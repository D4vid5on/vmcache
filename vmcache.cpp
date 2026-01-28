#include <atomic>
#include <algorithm>
#include <cassert>
#include <condition_variable>
#include <csignal>
#include <exception>
#include <fcntl.h>
#include <functional>
#include <iostream>
#include <mutex>
#include <numeric>
#include <set>
#include <thread>
#include <vector>
#include <span>

#include <errno.h>
#include <libaio.h>
#include <sys/mman.h>
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>
#include <immintrin.h>
#include <map>
#include <queue>
#include <shared_mutex>


#include "exmap.h"

__thread uint16_t workerThreadId = 0;
__thread int32_t tpcchistorycounter = 0;
#include "tpcc/TPCCWorkload.hpp"

using namespace std;

typedef uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;
typedef u64 PID; // page id type

static const u64 pageSize = 4096;

struct alignas(4096) Page {
   bool dirty;
};

static const int16_t maxWorkerThreads = 128;

#define die(msg) do { perror(msg); exit(EXIT_FAILURE); } while(0)

uint64_t rdtsc() {
   uint32_t hi, lo;
   __asm__ __volatile__ ("rdtsc" : "=a"(lo), "=d"(hi));
   return static_cast<uint64_t>(lo)|(static_cast<uint64_t>(hi)<<32);
}

// exmap helper function
static int exmapAction(int exmapfd, exmap_opcode op, u16 len) {
   struct exmap_action_params params_free = { .interface = workerThreadId, .iov_len = len, .opcode = (u16)op, };
   return ioctl(exmapfd, EXMAP_IOCTL_ACTION, &params_free);
}

// allocate memory using huge pages
void* allocHuge(size_t size) {
   void* p = mmap(NULL, size, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
   madvise(p, size, MADV_HUGEPAGE);
   return p;
}

// use when lock is not free
void yield(u64 counter) {
   _mm_pause();
}

struct PageState {
   atomic<u64> stateAndVersion;

   static const u64 Unlocked = 0;
   static const u64 MaxShared = 252;
   static const u64 Locked = 253;
   static const u64 Marked = 254;
   static const u64 Evicted = 255;

   PageState() {}

   void init() { stateAndVersion.store(sameVersion(0, Evicted), std::memory_order_release); }

   static inline u64 sameVersion(u64 oldStateAndVersion, u64 newState) { return ((oldStateAndVersion<<8)>>8) | newState<<56; }
   static inline u64 nextVersion(u64 oldStateAndVersion, u64 newState) { return (((oldStateAndVersion<<8)>>8)+1) | newState<<56; }

   bool tryLockX(u64 oldStateAndVersion) {
      return stateAndVersion.compare_exchange_strong(oldStateAndVersion, sameVersion(oldStateAndVersion, Locked));
   }

   void unlockX() {
      assert(getState() == Locked);
      stateAndVersion.store(nextVersion(stateAndVersion.load(), Unlocked), std::memory_order_release);
   }

   void unlockXEvicted() {
      assert(getState() == Locked);
      stateAndVersion.store(nextVersion(stateAndVersion.load(), Evicted), std::memory_order_release);
   }

   void downgradeLock() {
      assert(getState() == Locked);
      stateAndVersion.store(nextVersion(stateAndVersion.load(), 1), std::memory_order_release);
   }

   bool tryLockS(u64 oldStateAndVersion) {
      u64 s = getState(oldStateAndVersion);
      if (s<MaxShared)
         return stateAndVersion.compare_exchange_strong(oldStateAndVersion, sameVersion(oldStateAndVersion, s+1));
      if (s==Marked)
         return stateAndVersion.compare_exchange_strong(oldStateAndVersion, sameVersion(oldStateAndVersion, 1));
      return false;
   }

   void unlockS() {
      while (true) {
         u64 oldStateAndVersion = stateAndVersion.load();
         u64 state = getState(oldStateAndVersion);
         assert(state>0 && state<=MaxShared);
         if (stateAndVersion.compare_exchange_strong(oldStateAndVersion, sameVersion(oldStateAndVersion, state-1)))
            return;
      }
   }

   bool tryMark(u64 oldStateAndVersion) {
      assert(getState(oldStateAndVersion)==Unlocked);
      return stateAndVersion.compare_exchange_strong(oldStateAndVersion, sameVersion(oldStateAndVersion, Marked));
   }

   static u64 getState(u64 v) { return v >> 56; };
   u64 getState() { return getState(stateAndVersion.load()); }

   void operator=(PageState&) = delete;
};

// open addressing hash table used for second chance replacement to keep track of currently-cached pages
struct ResidentPageSet {
   static const u64 empty = ~0ull;
   static const u64 tombstone = (~0ull)-1;

   struct Entry {
      atomic<u64> pid;
   };

   Entry* ht;
   u64 count;
   u64 mask;
   atomic<u64> clockPos;

   ResidentPageSet(u64 maxCount) : count(next_pow2(maxCount * 1.5)), mask(count - 1), clockPos(0) {
      ht = (Entry*)allocHuge(count * sizeof(Entry));
      memset((void*)ht, 0xFF, count * sizeof(Entry));
   }

   ~ResidentPageSet() {
      munmap(ht, count * sizeof(u64));
   }

   u64 next_pow2(u64 x) {
      return 1<<(64-__builtin_clzl(x-1));
   }

   u64 hash(u64 k) {
      const u64 m = 0xc6a4a7935bd1e995;
      const int r = 47;
      u64 h = 0x8445d61a4e774912 ^ (8*m);
      k *= m;
      k ^= k >> r;
      k *= m;
      h ^= k;
      h *= m;
      h ^= h >> r;
      h *= m;
      h ^= h >> r;
      return h;
   }

   void insert(u64 pid) {
      u64 pos = hash(pid) & mask;
      while (true) {
         u64 curr = ht[pos].pid.load();
         assert(curr != pid);
         if ((curr == empty) || (curr == tombstone))
            if (ht[pos].pid.compare_exchange_strong(curr, pid))
               return;

         pos = (pos + 1) & mask;
      }
   }

   bool remove(u64 pid) {
      u64 pos = hash(pid) & mask;
      while (true) {
         u64 curr = ht[pos].pid.load();
         if (curr == empty)
            return false;

         if (curr == pid)
            if (ht[pos].pid.compare_exchange_strong(curr, tombstone))
               return true;

         pos = (pos + 1) & mask;
      }
   }

   template<class Fn>
   void iterateClockBatch(u64 batch, Fn fn) {
      u64 pos, newPos;
      do {
         pos = clockPos.load();
         newPos = (pos+batch) % count;
      } while (!clockPos.compare_exchange_strong(pos, newPos));

      for (u64 i=0; i<batch; i++) {
         u64 curr = ht[pos].pid.load();
         if ((curr != tombstone) && (curr != empty))
            fn(curr);
         pos = (pos + 1) & mask;
      }
   }
};

// libaio interface used to write batches of pages
struct LibaioInterface {
   static const u64 maxIOs = 256;

   int blockfd;
   Page* virtMem;
   io_context_t ctx;
   iocb cb[maxIOs];
   iocb* cbPtr[maxIOs];
   io_event events[maxIOs];

   LibaioInterface(int blockfd, Page* virtMem) : blockfd(blockfd), virtMem(virtMem) {
      memset(&ctx, 0, sizeof(io_context_t));
      int ret = io_setup(maxIOs, &ctx);
      if (ret != 0) {
         std::cerr << "libaio io_setup error: " << ret << " ";
         switch (-ret) {
            case EAGAIN: std::cerr << "EAGAIN"; break;
            case EFAULT: std::cerr << "EFAULT"; break;
            case EINVAL: std::cerr << "EINVAL"; break;
            case ENOMEM: std::cerr << "ENOMEM"; break;
            case ENOSYS: std::cerr << "ENOSYS"; break;
         };
         exit(EXIT_FAILURE);
      }
   }

   void writePages(const vector<PID>& pages) {
      assert(pages.size() < maxIOs);
      for (u64 i=0; i<pages.size(); i++) {
         PID pid = pages[i];
         virtMem[pid].dirty = false;
         cbPtr[i] = &cb[i];
         io_prep_pwrite(cb+i, blockfd, &virtMem[pid], pageSize, pageSize*pid);
      }
      int cnt = io_submit(ctx, pages.size(), cbPtr);
      assert(cnt == pages.size());
      cnt = io_getevents(ctx, pages.size(), pages.size(), events, nullptr);
      assert(cnt == pages.size());
   }
};

struct BufferManager {
   static const u64 mb = 1024ull * 1024;
   static const u64 gb = 1024ull * 1024 * 1024;
   u64 virtSize;
   u64 physSize;
   u64 virtCount;
   u64 physCount;
   struct exmap_user_interface* exmapInterface[maxWorkerThreads];
   vector<LibaioInterface> libaioInterface;

   bool useExmap;
   int blockfd;
   int exmapfd;

   atomic<u64> physUsedCount;
   ResidentPageSet residentSet;
   atomic<u64> allocCount;

   atomic<u64> readCount;
   atomic<u64> writeCount;
   atomic<u64> lsmSize;

   Page* virtMem;
   PageState* pageState;
   u64 batch;

   PageState& getPageState(PID pid) {
      return pageState[pid];
   }

   BufferManager();
   ~BufferManager() {}

   Page* fixX(PID pid);
   void unfixX(PID pid);
   Page* fixS(PID pid);
   void unfixS(PID pid);

   bool isValidPtr(void* page) { return (page >= virtMem) && (page < (virtMem + virtSize + 16)); }
   PID toPID(void* page) { return reinterpret_cast<Page*>(page) - virtMem; }
   Page* toPtr(PID pid) { return virtMem + pid; }

   void ensureFreePages();
   Page* allocPage();
   void handleFault(PID pid);
   void readPage(PID pid);
   void evict();
};


BufferManager bm;

struct OLCRestartException {};

template<class T>
struct GuardO {
   PID pid;
   T* ptr;
   u64 version;
   static const u64 moved = ~0ull;

   // constructor
   explicit GuardO(u64 pid) : pid(pid), ptr(reinterpret_cast<T*>(bm.toPtr(pid))) {
      init();
   }

   template<class T2>
   GuardO(u64 pid, GuardO<T2>& parent)  {
      parent.checkVersionAndRestart();
      this->pid = pid;
      ptr = reinterpret_cast<T*>(bm.toPtr(pid));
      init();
   }

   GuardO(GuardO&& other) {
      pid = other.pid;
      ptr = other.ptr;
      version = other.version;
   }

   void init() {
      assert(pid != moved);
      PageState& ps = bm.getPageState(pid);
      for (u64 repeatCounter=0; ; repeatCounter++) {
         u64 v = ps.stateAndVersion.load();
         switch (PageState::getState(v)) {
            case PageState::Marked: {
               u64 newV = PageState::sameVersion(v, PageState::Unlocked);
               if (ps.stateAndVersion.compare_exchange_weak(v, newV)) {
                  version = newV;
                  return;
               }
               break;
            }
            case PageState::Locked:
               break;
            case PageState::Evicted:
               if (ps.tryLockX(v)) {
                  bm.handleFault(pid);
                  bm.unfixX(pid);
               }
               break;
            default:
               version = v;
               return;
         }
         yield(repeatCounter);
      }
   }

   // move assignment operator
   GuardO& operator=(GuardO&& other) {
      if (pid != moved)
         checkVersionAndRestart();
      pid = other.pid;
      ptr = other.ptr;
      version = other.version;
      other.pid = moved;
      other.ptr = nullptr;
      return *this;
   }

   // assignment operator
   GuardO& operator=(const GuardO&) = delete;

   // copy constructor
   GuardO(const GuardO&) = delete;

   void checkVersionAndRestart() {
      if (pid != moved) {
         PageState& ps = bm.getPageState(pid);
         u64 stateAndVersion = ps.stateAndVersion.load();
         if (version == stateAndVersion) // fast path, nothing changed
            return;
         if ((stateAndVersion<<8) == (version<<8)) { // same version
            u64 state = PageState::getState(stateAndVersion);
            if (state <= PageState::MaxShared)
               return; // ignore shared locks
            if (state == PageState::Marked)
               if (ps.stateAndVersion.compare_exchange_weak(stateAndVersion, PageState::sameVersion(stateAndVersion, PageState::Unlocked)))
                  return; // mark cleared
         }
         if (std::uncaught_exceptions()==0)
            throw OLCRestartException();
      }
   }

   // destructor
   ~GuardO() noexcept(false) {
      checkVersionAndRestart();
   }

   T* operator->() {
      assert(pid != moved);
      return ptr;
   }

   void release() {
      checkVersionAndRestart();
      pid = moved;
      ptr = nullptr;
   }
};

template<class T>
struct GuardX {
   PID pid;
   T* ptr;
   static const u64 moved = ~0ull;

   // constructor
   GuardX(): pid(moved), ptr(nullptr) {}

   // constructor
   explicit GuardX(u64 pid) : pid(pid) {
      ptr = reinterpret_cast<T*>(bm.fixX(pid));
      ptr->dirty = true;
   }

   explicit GuardX(GuardO<T>&& other) {
      assert(other.pid != moved);
      for (u64 repeatCounter=0; ; repeatCounter++) {
         PageState& ps = bm.getPageState(other.pid);
         u64 stateAndVersion = ps.stateAndVersion;
         if ((stateAndVersion<<8) != (other.version<<8))
            throw OLCRestartException();
         u64 state = PageState::getState(stateAndVersion);
         if ((state == PageState::Unlocked) || (state == PageState::Marked)) {
            if (ps.tryLockX(stateAndVersion)) {
               pid = other.pid;
               ptr = other.ptr;
               ptr->dirty = true;
               other.pid = moved;
               other.ptr = nullptr;
               return;
            }
         }
         yield(repeatCounter);
      }
   }

   // assignment operator
   GuardX& operator=(const GuardX&) = delete;

   // move assignment operator
   GuardX& operator=(GuardX&& other) {
      if (pid != moved) {
         bm.unfixX(pid);
      }
      pid = other.pid;
      ptr = other.ptr;
      other.pid = moved;
      other.ptr = nullptr;
      return *this;
   }

   // copy constructor
   GuardX(const GuardX&) = delete;

   // destructor
   ~GuardX() {
      if (pid != moved)
         bm.unfixX(pid);
   }

   T* operator->() {
      assert(pid != moved);
      return ptr;
   }

   void release() {
      if (pid != moved) {
         bm.unfixX(pid);
         pid = moved;
      }
   }
};

template<class T>
struct AllocGuard : public GuardX<T> {
   template <typename ...Params>
   AllocGuard(Params&&... params) {
      GuardX<T>::ptr = reinterpret_cast<T*>(bm.allocPage());
      new (GuardX<T>::ptr) T(std::forward<Params>(params)...);
      GuardX<T>::pid = bm.toPID(GuardX<T>::ptr);
   }
};

template<class T>
struct GuardS {
   PID pid;
   T* ptr;
   static const u64 moved = ~0ull;

   // constructor
   explicit GuardS(u64 pid) : pid(pid) {
      ptr = reinterpret_cast<T*>(bm.fixS(pid));
   }

   GuardS(GuardO<T>&& other) {
      assert(other.pid != moved);
      if (bm.getPageState(other.pid).tryLockS(other.version)) { // XXX: optimize?
         pid = other.pid;
         ptr = other.ptr;
         other.pid = moved;
         other.ptr = nullptr;
      } else {
         throw OLCRestartException();
      }
   }

   GuardS(GuardS&& other) {
      if (pid != moved)
         bm.unfixS(pid);
      pid = other.pid;
      ptr = other.ptr;
      other.pid = moved;
      other.ptr = nullptr;
   }

   // assignment operator
   GuardS& operator=(const GuardS&) = delete;

   // move assignment operator
   GuardS& operator=(GuardS&& other) {
      if (pid != moved)
         bm.unfixS(pid);
      pid = other.pid;
      ptr = other.ptr;
      other.pid = moved;
      other.ptr = nullptr;
      return *this;
   }

   // copy constructor
   GuardS(const GuardS&) = delete;

   // destructor
   ~GuardS() {
      if (pid != moved)
         bm.unfixS(pid);
   }

   T* operator->() {
      assert(pid != moved);
      return ptr;
   }

   void release() {
      if (pid != moved) {
         bm.unfixS(pid);
         pid = moved;
      }
   }
};

u64 envOr(const char* env, u64 value) {
   if (getenv(env))
      return atof(getenv(env));
   return value;
}

BufferManager::BufferManager() : virtSize(envOr("VIRTGB", 16)*gb), physSize(envOr("PHYSGB", 4)*gb), virtCount(virtSize / pageSize), physCount(physSize / pageSize), residentSet(physCount) {
   assert(virtSize>=physSize);
   const char* path = getenv("BLOCK") ? getenv("BLOCK") : "/tmp/bm";
   blockfd = open(path, O_RDWR | O_DIRECT, S_IRWXU);
   if (blockfd == -1) {
      cerr << "cannot open BLOCK device '" << path << "'" << endl;
      exit(EXIT_FAILURE);
   }
   u64 virtAllocSize = virtSize + (1<<16); // we allocate 64KB extra to prevent segfaults during optimistic reads

   useExmap = envOr("EXMAP", 0);
   if (useExmap) {
      exmapfd = open("/dev/exmap", O_RDWR);
      if (exmapfd < 0) die("open exmap");

      struct exmap_ioctl_setup buffer;
      buffer.fd             = blockfd;
      buffer.max_interfaces = maxWorkerThreads;
      buffer.buffer_size    = physCount;
      buffer.flags          = 0;
      if (ioctl(exmapfd, EXMAP_IOCTL_SETUP, &buffer) < 0)
         die("ioctl: exmap_setup");

      for (unsigned i=0; i<maxWorkerThreads; i++) {
         exmapInterface[i] = (struct exmap_user_interface *) mmap(NULL, pageSize, PROT_READ|PROT_WRITE, MAP_SHARED, exmapfd, EXMAP_OFF_INTERFACE(i));
         if (exmapInterface[i] == MAP_FAILED)
            die("setup exmapInterface");
      }

      virtMem = (Page*)mmap(NULL, virtAllocSize, PROT_READ|PROT_WRITE, MAP_SHARED, exmapfd, 0);
   } else {
      virtMem = (Page*)mmap(NULL, virtAllocSize, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
      madvise(virtMem, virtAllocSize, MADV_NOHUGEPAGE);
   }

   pageState = (PageState*)allocHuge(virtCount * sizeof(PageState));
   for (u64 i=0; i<virtCount; i++)
      pageState[i].init();
   if (virtMem == MAP_FAILED)
      die("mmap failed");

   libaioInterface.reserve(maxWorkerThreads);
   for (unsigned i=0; i<maxWorkerThreads; i++)
      libaioInterface.emplace_back(LibaioInterface(blockfd, virtMem));

   physUsedCount = 0;
   allocCount = 1; // pid 0 reserved for meta data
   readCount = 0;
   writeCount = 0;
   lsmSize = 0;
   batch = envOr("BATCH", 64);

   cerr << "vmcache " << "blk:" << path << " virtgb:" << virtSize/gb << " physgb:" << physSize/gb << " exmap:" << useExmap << endl;
}

void BufferManager::ensureFreePages() {
   if (physUsedCount >= physCount*0.95)
      evict();
}

// allocated new page and fix it
Page* BufferManager::allocPage() {
   physUsedCount++;
   ensureFreePages();
   u64 pid = allocCount++;
   if (pid >= virtCount) {
      cerr << "VIRTGB is too low" << endl;
      exit(EXIT_FAILURE);
   }
   u64 stateAndVersion = getPageState(pid).stateAndVersion;
   bool succ = getPageState(pid).tryLockX(stateAndVersion);
   assert(succ);
   residentSet.insert(pid);

   if (useExmap) {
      exmapInterface[workerThreadId]->iov[0].page = pid;
      exmapInterface[workerThreadId]->iov[0].len = 1;
      while (exmapAction(exmapfd, EXMAP_OP_ALLOC, 1) < 0) {
         cerr << "allocPage errno: " << errno << " pid: " << pid << " workerId: " << workerThreadId << endl;
         ensureFreePages();
      }
   }
   virtMem[pid].dirty = true;

   return virtMem + pid;
}

void BufferManager::handleFault(PID pid) {
   physUsedCount++;
   ensureFreePages();
   readPage(pid);
   residentSet.insert(pid);
}

Page* BufferManager::fixX(PID pid) {
   PageState& ps = getPageState(pid);
   for (u64 repeatCounter=0; ; repeatCounter++) {
      u64 stateAndVersion = ps.stateAndVersion.load();
      switch (PageState::getState(stateAndVersion)) {
         case PageState::Evicted: {
            if (ps.tryLockX(stateAndVersion)) {
               handleFault(pid);
               return virtMem + pid;
            }
            break;
         }
         case PageState::Marked: case PageState::Unlocked: {
            if (ps.tryLockX(stateAndVersion))
               return virtMem + pid;
            break;
         }
      }
      yield(repeatCounter);
   }
}

Page* BufferManager::fixS(PID pid) {
   PageState& ps = getPageState(pid);
   for (u64 repeatCounter=0; ; repeatCounter++) {
      u64 stateAndVersion = ps.stateAndVersion;
      switch (PageState::getState(stateAndVersion)) {
         case PageState::Locked: {
            break;
         } case PageState::Evicted: {
            if (ps.tryLockX(stateAndVersion)) {
               handleFault(pid);
               ps.unlockX();
            }
            break;
         }
         default: {
            if (ps.tryLockS(stateAndVersion))
               return virtMem + pid;
         }
      }
      yield(repeatCounter);
   }
}

void BufferManager::unfixS(PID pid) {
   getPageState(pid).unlockS();
}

void BufferManager::unfixX(PID pid) {
   getPageState(pid).unlockX();
}

void BufferManager::readPage(PID pid) {
   if (useExmap) {
      for (u64 repeatCounter=0; ; repeatCounter++) {
         int ret = pread(exmapfd, virtMem+pid, pageSize, workerThreadId);
         if (ret == pageSize) {
            assert(ret == pageSize);
            readCount++;
            return;
         }
         cerr << "readPage errno: " << errno << " pid: " << pid << " workerId: " << workerThreadId << endl;
         ensureFreePages();
      }
   } else {
      int ret = pread(blockfd, virtMem+pid, pageSize, pid*pageSize);
      assert(ret==pageSize);
      readCount++;
   }
}

void BufferManager::evict() {
   vector<PID> toEvict;
   toEvict.reserve(batch);
   vector<PID> toWrite;
   toWrite.reserve(batch);

   // 0. find candidates, lock dirty ones in shared mode
   while (toEvict.size()+toWrite.size() < batch) {
      residentSet.iterateClockBatch(batch, [&](PID pid) {
         PageState& ps = getPageState(pid);
         u64 v = ps.stateAndVersion;
         switch (PageState::getState(v)) {
            case PageState::Marked:
               if (virtMem[pid].dirty) {
                  if (ps.tryLockS(v))
                     toWrite.push_back(pid);
               } else {
                  toEvict.push_back(pid);
               }
               break;
            case PageState::Unlocked:
               ps.tryMark(v);
               break;
            default:
               break; // skip
         };
      });
   }

   // 1. write dirty pages
   libaioInterface[workerThreadId].writePages(toWrite);
   writeCount += toWrite.size();

   // 2. try to lock clean page candidates
   toEvict.erase(std::remove_if(toEvict.begin(), toEvict.end(), [&](PID pid) {
      PageState& ps = getPageState(pid);
      u64 v = ps.stateAndVersion;
      return (PageState::getState(v) != PageState::Marked) || !ps.tryLockX(v);
   }), toEvict.end());

   // 3. try to upgrade lock for dirty page candidates
   for (auto& pid : toWrite) {
      PageState& ps = getPageState(pid);
      u64 v = ps.stateAndVersion;
      if ((PageState::getState(v) == 1) && ps.stateAndVersion.compare_exchange_weak(v, PageState::sameVersion(v, PageState::Locked)))
         toEvict.push_back(pid);
      else
         ps.unlockS();
   }

   // 4. remove from page table
   if (useExmap) {
      for (u64 i=0; i<toEvict.size(); i++) {
         exmapInterface[workerThreadId]->iov[i].page = toEvict[i];
         exmapInterface[workerThreadId]->iov[i].len = 1;
      }
      if (exmapAction(exmapfd, EXMAP_OP_FREE, toEvict.size()) < 0)
         die("ioctl: EXMAP_OP_FREE");
   } else {
      for (u64& pid : toEvict)
         madvise(virtMem + pid, pageSize, MADV_DONTNEED);
   }

   // 5. remove from hash table and unlock
   for (u64& pid : toEvict) {
      bool succ = residentSet.remove(pid);
      assert(succ);
      getPageState(pid).unlockXEvicted();
   }

   physUsedCount -= toEvict.size();
}

//---------------------------------------------------------------------------

struct BTreeNode;

struct BTreeNodeHeader {
   static const unsigned underFullSize = (pageSize/2) + (pageSize/4);  // merge nodes more empty
   static const u64 noNeighbour = ~0ull;

   struct FenceKeySlot {
      u16 offset;
      u16 len;
   };

   bool dirty;
   union {
      PID upperInnerNode; // inner
      PID nextLeafNode = noNeighbour; // leaf
   };

   bool hasRightNeighbour() { return nextLeafNode != noNeighbour; }

   FenceKeySlot lowerFence = {0, 0};  // exclusive
   FenceKeySlot upperFence = {0, 0};  // inclusive

   bool hasLowerFence() { return !!lowerFence.len; };

   u16 count = 0;
   bool isLeaf;
   u16 spaceUsed = 0;
   u16 dataOffset = static_cast<u16>(pageSize);
   u16 prefixLen = 0;

   static const unsigned hintCount = 16;
   u32 hint[hintCount];
   u32 padding;

   BTreeNodeHeader(bool isLeaf) : isLeaf(isLeaf) {}
   ~BTreeNodeHeader() {}
};

static unsigned min(unsigned a, unsigned b)
{
   return a < b ? a : b;
}

template <class T>
static T loadUnaligned(void* p)
{
   T x;
   memcpy(&x, p, sizeof(T));
   return x;
}

// Get order-preserving head of key (assuming little endian)
static u32 head(u8* key, unsigned keyLen)
{
   switch (keyLen) {
      case 0:
         return 0;
      case 1:
         return static_cast<u32>(key[0]) << 24;
      case 2:
         return static_cast<u32>(__builtin_bswap16(loadUnaligned<u16>(key))) << 16;
      case 3:
         return (static_cast<u32>(__builtin_bswap16(loadUnaligned<u16>(key))) << 16) | (static_cast<u32>(key[2]) << 8);
      default:
         return __builtin_bswap32(loadUnaligned<u32>(key));
   }
}

struct BTreeNode : public BTreeNodeHeader {
   struct Slot {
      u16 offset;
      u16 keyLen;
      u16 payloadLen;
      union {
         u32 head;
         u8 headBytes[4];
      };
   } __attribute__((packed));
   union {
      Slot slot[(pageSize - sizeof(BTreeNodeHeader)) / sizeof(Slot)];  // grows from front
      u8 heap[pageSize - sizeof(BTreeNodeHeader)];                // grows from back
   };

   static constexpr unsigned maxKVSize = ((pageSize - sizeof(BTreeNodeHeader) - (2 * sizeof(Slot)))) / 4;

   BTreeNode(bool isLeaf) : BTreeNodeHeader(isLeaf) { dirty = true; }

   u8* ptr() { return reinterpret_cast<u8*>(this); }
   bool isInner() { return !isLeaf; }
   span<u8> getLowerFence() { return { ptr() + lowerFence.offset, lowerFence.len}; }
   span<u8> getUpperFence() { return { ptr() + upperFence.offset, upperFence.len}; }
   u8* getPrefix() { return ptr() + lowerFence.offset; } // any key on page is ok

   unsigned freeSpace() { return dataOffset - (reinterpret_cast<u8*>(slot + count) - ptr()); }
   unsigned freeSpaceAfterCompaction() { return pageSize - (reinterpret_cast<u8*>(slot + count) - ptr()) - spaceUsed; }

   bool hasSpaceFor(unsigned keyLen, unsigned payloadLen)
   {
      return spaceNeeded(keyLen, payloadLen) <= freeSpaceAfterCompaction();
   }

   u8* getKey(unsigned slotId) { return ptr() + slot[slotId].offset; }
   span<u8> getPayload(unsigned slotId) { return {ptr() + slot[slotId].offset + slot[slotId].keyLen, slot[slotId].payloadLen}; }

   PID getChild(unsigned slotId) { return loadUnaligned<PID>(getPayload(slotId).data()); }

   // How much space would inserting a new key of len "keyLen" require?
   unsigned spaceNeeded(unsigned keyLen, unsigned payloadLen) {
      return sizeof(Slot) + (keyLen - prefixLen) + payloadLen;
   }

   void makeHint()
   {
      unsigned dist = count / (hintCount + 1);
      for (unsigned i = 0; i < hintCount; i++)
         hint[i] = slot[dist * (i + 1)].head;
   }

   void updateHint(unsigned slotId)
   {
      unsigned dist = count / (hintCount + 1);
      unsigned begin = 0;
      if ((count > hintCount * 2 + 1) && (((count - 1) / (hintCount + 1)) == dist) && ((slotId / dist) > 1))
         begin = (slotId / dist) - 1;
      for (unsigned i = begin; i < hintCount; i++)
         hint[i] = slot[dist * (i + 1)].head;
   }

   void searchHint(u32 keyHead, u16& lowerOut, u16& upperOut)
   {
      if (count > hintCount * 2) {
         u16 dist = upperOut / (hintCount + 1);
         u16 pos, pos2;
         for (pos = 0; pos < hintCount; pos++)
            if (hint[pos] >= keyHead)
               break;
         for (pos2 = pos; pos2 < hintCount; pos2++)
            if (hint[pos2] != keyHead)
               break;
         lowerOut = pos * dist;
         if (pos2 < hintCount)
            upperOut = (pos2 + 1) * dist;
      }
   }

   // lower bound search, foundExactOut indicates if there is an exact match, returns slotId
   u16 lowerBound(span<u8> skey, bool& foundExactOut)
   {
      foundExactOut = false;

      // check prefix
      int cmp = memcmp(skey.data(), getPrefix(), min(skey.size(), prefixLen));
      if (cmp < 0) // key is less than prefix
         return 0;
      if (cmp > 0) // key is greater than prefix
         return count;
      if (skey.size() < prefixLen) // key is equal but shorter than prefix
         return 0;
      u8* key = skey.data() + prefixLen;
      unsigned keyLen = skey.size() - prefixLen;

      // check hint
      u16 lower = 0;
      u16 upper = count;
      u32 keyHead = head(key, keyLen);
      searchHint(keyHead, lower, upper);

      // binary search on remaining range
      while (lower < upper) {
         u16 mid = ((upper - lower) / 2) + lower;
         if (keyHead < slot[mid].head) {
            upper = mid;
         } else if (keyHead > slot[mid].head) {
            lower = mid + 1;
         } else { // head is equal, check full key
            int cmp = memcmp(key, getKey(mid), min(keyLen, slot[mid].keyLen));
            if (cmp < 0) {
               upper = mid;
            } else if (cmp > 0) {
               lower = mid + 1;
            } else {
               if (keyLen < slot[mid].keyLen) { // key is shorter
                  upper = mid;
               } else if (keyLen > slot[mid].keyLen) { // key is longer
                  lower = mid + 1;
               } else {
                  foundExactOut = true;
                  return mid;
               }
            }
         }
      }
      return lower;
   }

   // lowerBound wrapper ignoring exact match argument (for convenience)
   u16 lowerBound(span<u8> key)
   {
      bool ignore;
      return lowerBound(key, ignore);
   }

   // insert key/value pair
   void insertInPage(span<u8> key, span<u8> payload)
   {
      unsigned needed = spaceNeeded(key.size(), payload.size());
      if (needed > freeSpace()) {
         assert(needed <= freeSpaceAfterCompaction());
         compactify();
      }
      unsigned slotId = lowerBound(key);
      memmove(slot + slotId + 1, slot + slotId, sizeof(Slot) * (count - slotId));
      storeKeyValue(slotId, key, payload);
      count++;
      updateHint(slotId);
   }

   bool removeSlot(unsigned slotId)
   {
      spaceUsed -= slot[slotId].keyLen;
      spaceUsed -= slot[slotId].payloadLen;
      memmove(slot + slotId, slot + slotId + 1, sizeof(Slot) * (count - slotId - 1));
      count--;
      makeHint();
      return true;
   }

   bool removeInPage(span<u8> key)
   {
      bool found;
      unsigned slotId = lowerBound(key, found);
      if (!found)
         return false;
      return removeSlot(slotId);
   }

   void copyNode(BTreeNodeHeader* dst, BTreeNodeHeader* src) {
      u64 ofs = offsetof(BTreeNodeHeader, upperInnerNode);
      memcpy(reinterpret_cast<u8*>(dst)+ofs, reinterpret_cast<u8*>(src)+ofs, sizeof(BTreeNode)-ofs);
   }

   void compactify()
   {
      unsigned should = freeSpaceAfterCompaction();
      static_cast<void>(should);
      BTreeNode tmp(isLeaf);
      tmp.setFences(getLowerFence(), getUpperFence());
      copyKeyValueRange(&tmp, 0, 0, count);
      tmp.upperInnerNode = upperInnerNode;
      copyNode(this, &tmp);
      makeHint();
      assert(freeSpace() == should);
   }

   // merge right node into this node
   bool mergeNodes(unsigned slotId, BTreeNode* parent, BTreeNode* right)
   {
      if (!isLeaf)
         // TODO: implement inner merge
         return true;

      assert(right->isLeaf);
      assert(parent->isInner());
      BTreeNode tmp(isLeaf);
      tmp.setFences(getLowerFence(), right->getUpperFence());
      unsigned leftGrow = (prefixLen - tmp.prefixLen) * count;
      unsigned rightGrow = (right->prefixLen - tmp.prefixLen) * right->count;
      unsigned spaceUpperBound =
         spaceUsed + right->spaceUsed + (reinterpret_cast<u8*>(slot + count + right->count) - ptr()) + leftGrow + rightGrow;
      if (spaceUpperBound > pageSize)
         return false;
      copyKeyValueRange(&tmp, 0, 0, count);
      right->copyKeyValueRange(&tmp, count, 0, right->count);
      PID pid = bm.toPID(this);
      memcpy(parent->getPayload(slotId+1).data(), &pid, sizeof(PID));
      parent->removeSlot(slotId);
      tmp.makeHint();
      tmp.nextLeafNode = right->nextLeafNode;

      copyNode(this, &tmp);
      return true;
   }

   // store key/value pair at slotId
   void storeKeyValue(u16 slotId, span<u8> skey, span<u8> payload)
   {
      // slot
      u8* key = skey.data() + prefixLen;
      unsigned keyLen = skey.size() - prefixLen;
      slot[slotId].head = head(key, keyLen);
      slot[slotId].keyLen = keyLen;
      slot[slotId].payloadLen = payload.size();
      // key
      unsigned space = keyLen + payload.size();
      dataOffset -= space;
      spaceUsed += space;
      slot[slotId].offset = dataOffset;
      assert(getKey(slotId) >= reinterpret_cast<u8*>(&slot[slotId]));
      memcpy(getKey(slotId), key, keyLen);
      memcpy(getPayload(slotId).data(), payload.data(), payload.size());
   }

   void copyKeyValueRange(BTreeNode* dst, u16 dstSlot, u16 srcSlot, unsigned srcCount)
   {
      if (prefixLen <= dst->prefixLen) {  // prefix grows
         unsigned diff = dst->prefixLen - prefixLen;
         for (unsigned i = 0; i < srcCount; i++) {
            unsigned newKeyLen = slot[srcSlot + i].keyLen - diff;
            unsigned space = newKeyLen + slot[srcSlot + i].payloadLen;
            dst->dataOffset -= space;
            dst->spaceUsed += space;
            dst->slot[dstSlot + i].offset = dst->dataOffset;
            u8* key = getKey(srcSlot + i) + diff;
            memcpy(dst->getKey(dstSlot + i), key, space);
            dst->slot[dstSlot + i].head = head(key, newKeyLen);
            dst->slot[dstSlot + i].keyLen = newKeyLen;
            dst->slot[dstSlot + i].payloadLen = slot[srcSlot + i].payloadLen;
         }
      } else {
         for (unsigned i = 0; i < srcCount; i++)
            copyKeyValue(srcSlot + i, dst, dstSlot + i);
      }
      dst->count += srcCount;
      assert((dst->ptr() + dst->dataOffset) >= reinterpret_cast<u8*>(dst->slot + dst->count));
   }

   void copyKeyValue(u16 srcSlot, BTreeNode* dst, u16 dstSlot)
   {
      unsigned fullLen = slot[srcSlot].keyLen + prefixLen;
      u8 key[fullLen];
      memcpy(key, getPrefix(), prefixLen);
      memcpy(key+prefixLen, getKey(srcSlot), slot[srcSlot].keyLen);
      dst->storeKeyValue(dstSlot, {key, fullLen}, getPayload(srcSlot));
   }

   void insertFence(FenceKeySlot& fk, span<u8> key)
   {
      assert(freeSpace() >= key.size());
      dataOffset -= key.size();
      spaceUsed += key.size();
      fk.offset = dataOffset;
      fk.len = key.size();
      memcpy(ptr() + dataOffset, key.data(), key.size());
   }

   void setFences(span<u8> lower, span<u8> upper)
   {
      insertFence(lowerFence, lower);
      insertFence(upperFence, upper);
      for (prefixLen = 0; (prefixLen < min(lower.size(), upper.size())) && (lower[prefixLen] == upper[prefixLen]); prefixLen++)
         ;
   }

   void splitNode(BTreeNode* parent, unsigned sepSlot, span<u8> sep)
   {
      assert(sepSlot > 0);
      assert(sepSlot < (pageSize / sizeof(PID)));

      BTreeNode tmp(isLeaf);
      BTreeNode* nodeLeft = &tmp;

      AllocGuard<BTreeNode> newNode(isLeaf);
      BTreeNode* nodeRight = newNode.ptr;

      nodeLeft->setFences(getLowerFence(), sep);
      nodeRight->setFences(sep, getUpperFence());

      PID leftPID = bm.toPID(this);
      u16 oldParentSlot = parent->lowerBound(sep);
      if (oldParentSlot == parent->count) {
         assert(parent->upperInnerNode == leftPID);
         parent->upperInnerNode = newNode.pid;
      } else {
         assert(parent->getChild(oldParentSlot) == leftPID);
         memcpy(parent->getPayload(oldParentSlot).data(), &newNode.pid, sizeof(PID));
      }
      parent->insertInPage(sep, {reinterpret_cast<u8*>(&leftPID), sizeof(PID)});

      if (isLeaf) {
         copyKeyValueRange(nodeLeft, 0, 0, sepSlot + 1);
         copyKeyValueRange(nodeRight, 0, nodeLeft->count, count - nodeLeft->count);
         nodeLeft->nextLeafNode = newNode.pid;
         nodeRight->nextLeafNode = this->nextLeafNode;
      } else {
         // in inner node split, separator moves to parent (count == 1 + nodeLeft->count + nodeRight->count)
         copyKeyValueRange(nodeLeft, 0, 0, sepSlot);
         copyKeyValueRange(nodeRight, 0, nodeLeft->count + 1, count - nodeLeft->count - 1);
         nodeLeft->upperInnerNode = getChild(nodeLeft->count);
         nodeRight->upperInnerNode = upperInnerNode;
      }
      nodeLeft->makeHint();
      nodeRight->makeHint();
      copyNode(this, nodeLeft);
   }

   struct SeparatorInfo {
      unsigned len;      // len of new separator
      unsigned slot;     // slot at which we split
      bool isTruncated;  // if true, we truncate the separator taking len bytes from slot+1
   };

   unsigned commonPrefix(unsigned slotA, unsigned slotB)
   {
      assert(slotA < count);
      unsigned limit = min(slot[slotA].keyLen, slot[slotB].keyLen);
      u8 *a = getKey(slotA), *b = getKey(slotB);
      unsigned i;
      for (i = 0; i < limit; i++)
         if (a[i] != b[i])
            break;
      return i;
   }

   SeparatorInfo findSeparator(bool splitOrdered)
   {
      assert(count > 1);
      if (isInner()) {
         // inner nodes are split in the middle
         unsigned slotId = count / 2;
         return SeparatorInfo{static_cast<unsigned>(prefixLen + slot[slotId].keyLen), slotId, false};
      }

      // find good separator slot
      unsigned bestPrefixLen, bestSlot;

      if (splitOrdered) {
         bestSlot = count - 2;
      } else if (count > 16) {
         unsigned lower = (count / 2) - (count / 16);
         unsigned upper = (count / 2);

         bestPrefixLen = commonPrefix(lower, 0);
         bestSlot = lower;

         if (bestPrefixLen != commonPrefix(upper - 1, 0))
            for (bestSlot = lower + 1; (bestSlot < upper) && (commonPrefix(bestSlot, 0) == bestPrefixLen); bestSlot++)
               ;
      } else {
         bestSlot = (count-1) / 2;
      }


      // try to truncate separator
      unsigned common = commonPrefix(bestSlot, bestSlot + 1);
      if ((bestSlot + 1 < count) && (slot[bestSlot].keyLen > common) && (slot[bestSlot + 1].keyLen > (common + 1)))
         return SeparatorInfo{prefixLen + common + 1, bestSlot, true};

      return SeparatorInfo{static_cast<unsigned>(prefixLen + slot[bestSlot].keyLen), bestSlot, false};
   }

   void getSep(u8* sepKeyOut, SeparatorInfo info)
   {
      memcpy(sepKeyOut, getPrefix(), prefixLen);
      memcpy(sepKeyOut + prefixLen, getKey(info.slot + info.isTruncated), info.len - prefixLen);
   }

   PID lookupInner(span<u8> key)
   {
      unsigned pos = lowerBound(key);
      if (pos == count)
         return upperInnerNode;
      return getChild(pos);
   }
};


struct Level0Entry  // store min and max key for each file so we don't have to open them to check if the key can be inside
{                   // basically a manifest entry cause all level entryies are the same
   u64 fileID;
   u64 minKey;
   u64 maxKey;
};

struct LsmRootPage   // max 4096 bytes, after that it does not fit on one page, could split into multiple pages for more capacity
{                    // level0entry has size 24B so we can fit 168 file entries, set it to 150, so first level has 10 and size ratio is 2
   bool dirty;
   u64 nextSSTableId;
   static constexpr u32 L0_COMPACTION_TRIGGER = 4;
   static constexpr u32 MAX_L0_FILES = 10;
   static constexpr u32 MAX_L1_FILES = 20;
   static constexpr u32 MAX_L2_FILES = 40;
   static constexpr u32 MAX_L3_FILES = 80;
   u32 l0_count;
   Level0Entry l0Entries[MAX_L0_FILES];  // overlapping key ranges
   u32 l1_count;
   Level0Entry l1Entries[MAX_L1_FILES];  // non-overlapping
   u32 l2_count;
   Level0Entry l2Entries[MAX_L2_FILES];  // non-overlapping
   u32 l3_count;
   Level0Entry l3Entries[MAX_L3_FILES];  // non-overlapping
   LsmRootPage()
   {
      memset(this, 0, pageSize);
      nextSSTableId = 1;
      l0_count = 0;
      l1_count = 0;
      l2_count = 0;
      l3_count = 0;
      dirty = true;
   };
};


static_assert(sizeof(BTreeNode) == pageSize, "btree node size problem");

static const u64 metadataPageId = 0;

struct MetaDataPage {
   bool dirty;
   PID roots[(pageSize-8)/8];
   PID lsmRootPid;
   PID getRoot(unsigned slot) { return roots[slot]; }
};

struct BTree {
   private:

   void trySplit(GuardX<BTreeNode>&& node, GuardX<BTreeNode>&& parent, span<u8> key, unsigned payloadLen);
   void ensureSpace(BTreeNode* toSplit, span<u8> key, unsigned payloadLen);

   public:
   unsigned slotId;
   atomic<bool> splitOrdered;

   BTree();
   ~BTree();

   GuardO<BTreeNode> findLeafO(span<u8> key) {
      GuardO<MetaDataPage> meta(metadataPageId);
      GuardO<BTreeNode> node(meta->getRoot(slotId), meta);
      meta.release();

      while (node->isInner())
         node = GuardO<BTreeNode>(node->lookupInner(key), node);
      return node;
   }

   // point lookup, returns payload len on success, or -1 on failure
   int lookup(span<u8> key, u8* payloadOut, unsigned payloadOutSize) {
      for (u64 repeatCounter=0; ; repeatCounter++) {
         try {
            GuardO<BTreeNode> node = findLeafO(key);
            bool found;
            unsigned pos = node->lowerBound(key, found);
            if (!found)
               return -1;

            // key found, copy payload
            memcpy(payloadOut, node->getPayload(pos).data(), min(node->slot[pos].payloadLen, payloadOutSize));
            return node->slot[pos].payloadLen;
         } catch(const OLCRestartException&) { yield(repeatCounter); }
      }
   }

   template<class Fn>
   bool lookup(span<u8> key, Fn fn) {
      for (u64 repeatCounter=0; ; repeatCounter++) {
         try {
            GuardO<BTreeNode> node = findLeafO(key);
            bool found;
            unsigned pos = node->lowerBound(key, found);
            if (!found)
               return false;

            // key found
            fn(node->getPayload(pos));
            return true;
         } catch(const OLCRestartException&) { yield(repeatCounter); }
      }
   }

   void insert(span<u8> key, span<u8> payload);
   bool remove(span<u8> key);

   template<class Fn>
   bool updateInPlace(span<u8> key, Fn fn) {
      for (u64 repeatCounter=0; ; repeatCounter++) {
         try {
            GuardO<BTreeNode> node = findLeafO(key);
            bool found;
            unsigned pos = node->lowerBound(key, found);
            if (!found)
               return false;

            {
               GuardX<BTreeNode> nodeLocked(move(node));
               fn(nodeLocked->getPayload(pos));
               return true;
            }
         } catch(const OLCRestartException&) { yield(repeatCounter); }
      }
   }

   GuardS<BTreeNode> findLeafS(span<u8> key) {
      for (u64 repeatCounter=0; ; repeatCounter++) {
         try {
            GuardO<MetaDataPage> meta(metadataPageId);
            GuardO<BTreeNode> node(meta->getRoot(slotId), meta);
            meta.release();

            while (node->isInner())
               node = GuardO<BTreeNode>(node->lookupInner(key), node);

            return GuardS<BTreeNode>(move(node));
         } catch(const OLCRestartException&) { yield(repeatCounter); }
      }
   }

   template<class Fn>
   void scanAsc(span<u8> key, Fn fn) {     // modified to use fn with span, like lsm tree
      GuardS<BTreeNode> node = findLeafS(key);
      bool found;
      unsigned pos = node->lowerBound(key, found);
      u8 keyBuffer[pageSize];
      for (u64 repeatCounter=0; ; repeatCounter++) { // XXX
         if (pos<node->count) {
            unsigned len = node->prefixLen + node->slot[pos].keyLen;
            memcpy(keyBuffer, node->getPrefix(), node->prefixLen);
            memcpy(keyBuffer + node->prefixLen, node->getKey(pos), node->slot[pos].keyLen);
            span<u8> keySpan(keyBuffer, len);
            span<u8> payloadSpan = node->getPayload(pos);
            if (!fn(keySpan, payloadSpan))
               return;
            pos++;
         } else {
            if (!node->hasRightNeighbour())
               return;
            pos = 0;
            node = GuardS<BTreeNode>(node->nextLeafNode);
         }
      }
   }

   template<class Fn>
   void scanDesc(span<u8> key, Fn fn) {    // also modified so it matches lsm tree
      GuardS<BTreeNode> node = findLeafS(key);
      bool exactMatch;
      int pos = node->lowerBound(key, exactMatch);
      if (pos == node->count) {
         pos--;
         exactMatch = true; // XXX:
      }
      u8 keyBuffer[pageSize];
      for (u64 repeatCounter=0; ; repeatCounter++) { // XXX
         while (pos>=0) {
            unsigned len = node->prefixLen + node->slot[pos].keyLen;
            memcpy(keyBuffer, node->getPrefix(), node->prefixLen);
            memcpy(keyBuffer + node->prefixLen, node->getKey(pos), node->slot[pos].keyLen);
            span<u8> keySpan(keyBuffer, len);
            span<u8> payloadSpan = node->getPayload(pos);
            if (!fn(keySpan, payloadSpan))
               return;
            pos--;
         }
         if (!node->hasLowerFence())
            return;
         node = findLeafS(node->getLowerFence());
         pos = node->count-1;
      }
   }
};





struct MemtableEntry
{
   string payload;
   bool isTombstone = false;
   MemtableEntry(span<u8> payload_span, bool tombstone = false) :
   payload(reinterpret_cast<const char*>(payload_span.data()), payload_span.size()), isTombstone(tombstone) {};
   MemtableEntry() = default;
};
using Memtable = map<string, MemtableEntry>;   // we use a map as the memtable since it stores data sorted after the keys

struct SSTableMetadata   // we will store metadata of an sstable on the first block of the sstable, so we know how to read the file
{
   u64 fileID;
   u32 level;
   string minKey;   // to know the key ranges of an sstable for lookup
   string maxKey;
   u64 blockIndexOffset;  // the offset of where the indexes are and how many they are
   u32 numBlocks;

};
constexpr u32 SSTABLE_BLOCK_SIZE = 4096;  // same as page size

struct BlockIndexEntry
{
   string lastKey;   // store the last key of each block (the biggest)
   u64 blockOffset;  // and where it starts
};

class Serializer   // helper to serialize data
{
private:
   u8* buffer;
   u64 offset;
public:
   Serializer (u8* buf, u64 initial_off = 0) : buffer(buf), offset(initial_off) {}
   u64 getOffset() const { return offset; }
   void write (const auto& value)
   {
      memcpy(buffer+offset, &value, sizeof(value));
      offset += sizeof(value);
   }
   void writeKey(const string& key)
   {
      u16 len = key.size();
      write(len);
      memcpy(buffer+offset, key.data(), len);
      offset += len;
   }
   void writeSpan(span<u8> data)
   {
      memcpy(buffer+offset, data.data(), data.size());
      offset += data.size();
   }
};

class SSTableWriter
{
private:
   u64 fileFD;
   u64 fileID;
   u64 currentOffset;
   vector<u8> blockBuffer;
   vector<BlockIndexEntry> blockIndex;
   string lastKeyInBlock = "";
   string firstKeyInTable = "";
   u64 currentBlockSize = 0;
   u64 blockOffset1 = SSTABLE_BLOCK_SIZE;  // first block is for metadata so we don't start from 0
   void writeBlock(const u8* data, u64 size);
   void writeIndexAndMetadata(SSTableMetadata& metadata);
   u64 getRecordSize(const string& key, const MemtableEntry& entry);
   u64 serializeRecord(u8* buffer, u64 currentPosition, const string& key, const MemtableEntry& entry);
public:
   explicit SSTableWriter (u64 FileID, const string& directory = "/tmp/sstables")    // all sstables will be stored in /tmp/sstables and will be called fileId.sst
      : currentOffset(SSTABLE_BLOCK_SIZE)                                               // fileId will be unique
   {
      fileID = FileID;
      string path = directory + "/" + to_string(FileID) + ".sst";
      mkdir(directory.c_str(), 0777);
      fileFD = open(path.c_str(), O_RDWR | O_CREAT | O_TRUNC | O_DIRECT, 0644);  // O_DIRECT to bypass OS page cache,
      if (fileFD < 0)                                                                     // o_trunc so in case we use the same id more times we just wipe the older file (should not happen, but to be safe)
      {
         throw runtime_error("Failed to  open (for writer) sstable file " + path);
      }
      blockBuffer.resize(SSTABLE_BLOCK_SIZE);
      currentOffset = SSTABLE_BLOCK_SIZE;
   }
   ~SSTableWriter()
   {
      if (fileFD >= 0) close(fileFD);
   }
   u64 getCurrentSize() const
   {
      return currentOffset + currentBlockSize;
   }
   SSTableMetadata writeMemtable(Memtable& memtable, u8 targetLevel);
   void addRecord(string_view key, string_view payload, bool isTombstone);
   void finish(SSTableMetadata& metadata);
};
// get size of memtable entry including key
u64 SSTableWriter::getRecordSize(const string& key, const MemtableEntry& entry)
{
   return sizeof(u16) + key.size() + sizeof(u16) + entry.payload.size() + sizeof(u8);
}
   // serialize a memtableEntry in the buffer at the given position
u64 SSTableWriter::serializeRecord(u8* buffer, u64 currentPosition, const string& key, const MemtableEntry& entry)
{
   u8* start = buffer + currentPosition;
   u8* ptr = start;
   u16 keyLen = key.size();
   u16 payloadLen = entry.payload.size();
   u8 isTombstone = entry.isTombstone;
   memcpy(ptr, &keyLen, sizeof(keyLen));  //key
   ptr += sizeof(keyLen);
   memcpy(ptr, key.data(), keyLen);
   ptr += keyLen;
   memcpy(ptr, &payloadLen, sizeof(u16) );   //payload
   ptr += sizeof(u16);
   memcpy(ptr, entry.payload.data(), payloadLen);
   ptr += payloadLen;
   memcpy(ptr, &isTombstone, sizeof(u8));
   ptr += sizeof(u8);
   return static_cast<u64>(ptr - start);   // return size of the write
}
   // write a block to the file (to disk)
void SSTableWriter::writeBlock(const u8* data, u64 size)
{
   assert(currentOffset % SSTABLE_BLOCK_SIZE == 0);
   vector<u8> paddedBlock(SSTABLE_BLOCK_SIZE, 0);
   memcpy(paddedBlock.data(), data, size);
   ssize_t result = pwrite(fileFD, paddedBlock.data(), SSTABLE_BLOCK_SIZE, currentOffset);
   assert(result == SSTABLE_BLOCK_SIZE);
   bm.writeCount.fetch_add(1);  // for wmb benchmark
   bm.lsmSize.fetch_add(SSTABLE_BLOCK_SIZE); // for size benchmark
   currentOffset += SSTABLE_BLOCK_SIZE;
}
  // writes index entries at the end of the file and metadata of the sstable in the first reserve block (offset 0)
void SSTableWriter::writeIndexAndMetadata(SSTableMetadata& metadata)
{
   metadata.blockIndexOffset = currentOffset;
   metadata.numBlocks = blockIndex.size();
   u64 indexSize = 0;
   fill(blockBuffer.begin(), blockBuffer.end(), 0);
   for (const auto& entry : blockIndex) // write the index to disk, as many entries per block as we can fit
   {
      if (indexSize + getRecordSize(entry.lastKey, {}) > SSTABLE_BLOCK_SIZE) //if next entry doesn't fit, write the buffer to disk
      {
         writeBlock(blockBuffer.data(), indexSize);
         indexSize = 0;
         fill(blockBuffer.begin(), blockBuffer.end(), 0);
      }
      u8* ptr = blockBuffer.data() + indexSize;
      u16 keyLen = entry.lastKey.size();
      memcpy(ptr, &keyLen, sizeof(u16));
      ptr += sizeof(u16);
      memcpy(ptr, entry.lastKey.data(), keyLen);
      ptr += keyLen;
      memcpy(ptr, &entry.blockOffset, sizeof(u64));
      ptr += sizeof(u64);
      indexSize += sizeof(u64) + sizeof(u16) + keyLen;
   }
   if (indexSize > 0) // write last block to disk
   {
      writeBlock(blockBuffer.data(), indexSize);
   }
   fill(blockBuffer.begin(), blockBuffer.end(), 0);
   Serializer metaSerializer(blockBuffer.data()); // write metadata
   metaSerializer.write(metadata.fileID);
   metaSerializer.write(metadata.level);
   metaSerializer.writeKey(metadata.minKey);
   metaSerializer.writeKey(metadata.maxKey);
   metaSerializer.write(metadata.blockIndexOffset);
   metaSerializer.write(metadata.numBlocks);
   ssize_t result = pwrite(fileFD, blockBuffer.data(), SSTABLE_BLOCK_SIZE, 0); // write it to disk at the beginning of the file
   assert(result == SSTABLE_BLOCK_SIZE);
   bm.writeCount.fetch_add(1);   // for wmb benchmark
   bm.lsmSize.fetch_add(SSTABLE_BLOCK_SIZE);  // for size benchmark
   close(fileFD);
}

class SSTableReader
{
private:
   int fd;
   SSTableMetadata metadata;
   vector<BlockIndexEntry> index;
   u8* blockBuffer;

public:
   SSTableReader(u64 fileID, const string& dir = "/tmp/sstables" )  // opens file fileID to read from it
   {
      string path = dir + "/" + to_string(fileID) + ".sst";
      fd = open(path.c_str(), O_RDONLY | O_DIRECT);
      if (fd < 0) throw std::runtime_error("Failed to open (for reader) sst file " + path);
      blockBuffer = (u8*)aligned_alloc(4096, 4096);
      if (pread(fd, blockBuffer, 4096, 0) != 4096) throw std::runtime_error("Failed to read sst file " + path); // read first block (metadata)
      bm.readCount.fetch_add(1);  // for rmb benchmark
      u8* ptr = blockBuffer;
      memcpy(&metadata.fileID, ptr, sizeof(u64)); ptr += sizeof(u64); // fill metadata with the metadata from file
      memcpy(&metadata.level, ptr, sizeof(u32)); ptr += sizeof(u32);
      auto readString = [&](string& str)
      {
         u16 len;
         memcpy(&len, ptr, sizeof(u16)); ptr += sizeof(u16);  // read length and store it in len
         str.assign((char*) ptr, len); ptr += len;  // read the string of length len and store it in the referance parameter
      };
      readString(metadata.minKey);
      readString(metadata.maxKey);
      memcpy(&metadata.blockIndexOffset, ptr, sizeof(u64)); ptr += sizeof(u64);
      memcpy(&metadata.numBlocks, ptr, sizeof(u32)); ptr += sizeof(u32);
      loadIndex();  // read index
   }
   ~SSTableReader()
   {
      if (fd >= 0) close(fd);
      free(blockBuffer);
   }
   void loadIndex();
   template<class Fn>
   int lookup(span<u8> key, Fn fn)       // returns 0 if not in sstable, 1 if in sstable but is tombstone, 2 if in sstable and not tombstone
   { // efficient binary search for the first block in which key could be
      auto it = lower_bound(index.begin(), index.end(), key, [](const BlockIndexEntry& entry, span<u8> k)
      {    // it points to the first element bigger than key, because we need the lastKey of the block to be bigger than the key since sstable are sorted
         string_view entryKey(entry.lastKey);
         string_view searchKey((char*)k.data(), k.size());
         return entryKey < searchKey;  // this comparator is used to eliminate the invalid options, that is why it looks like it is the wrong way around
      });
      if (it == index.end()) return 0;
      if (pread(fd, blockBuffer, SSTABLE_BLOCK_SIZE, it->blockOffset) != SSTABLE_BLOCK_SIZE) return 0; // read the candidate block
      bm.readCount.fetch_add(1);  // for rmb benchmark
      u64 offset = 0;
      while (offset < SSTABLE_BLOCK_SIZE)  // blocks are small, we just iterate through until we find key
      {
         u16 keyLen;   // read key, payload and isTombstone
         memcpy(&keyLen, blockBuffer + offset, sizeof(u16));
         if (keyLen == 0) break;
         u8* keyPtr = blockBuffer + offset + sizeof(u16);
         u16 payloadLen;
         memcpy(&payloadLen, keyPtr + keyLen, sizeof(u16));
         u8* payloadPtr = keyPtr + keyLen + sizeof(u16);
         u8 isTombstone = *(payloadPtr + payloadLen);
         if (keyLen == key.size() && memcmp(keyPtr, key.data(), keyLen) == 0)
         {
            if (isTombstone) return 1;  // it has been deleted but we did find it
            fn(span<u8>(payloadPtr, payloadLen));
            return 2;
         }
         offset += sizeof(u16) + keyLen + sizeof(u16) + payloadLen + sizeof(u8);
      }
      return 0;
   }
   int const getFd() const {return fd;}
   const SSTableMetadata& getMetadata() const {return metadata;}
   u64 const getBlockOffset(u64 blockIndex) const
   {
      if (blockIndex >= index.size()) return 0;
      return index[blockIndex].blockOffset;
   }
};

void SSTableReader::loadIndex()
{
   u64 currentOffset = metadata.blockIndexOffset;
   while (index.size() < metadata.numBlocks )   // read 4kb chunks until there are no more blockIndexEntries
   {
      if (pread(fd, blockBuffer, SSTABLE_BLOCK_SIZE, currentOffset) != SSTABLE_BLOCK_SIZE) break;
      bm.readCount.fetch_add(1);  // for rmb benchmark;
      u64 bufferOffset = 0;
      while (bufferOffset < SSTABLE_BLOCK_SIZE && index.size() < metadata.numBlocks)  // add entries into index from the block we read
      {
         u16 len;
         memcpy(&len, blockBuffer + bufferOffset, sizeof(u16));
         bufferOffset += sizeof(u16);
         if (len == 0) break;
         if (bufferOffset + len + 2 + 8 > SSTABLE_BLOCK_SIZE) break;
         BlockIndexEntry entry;
         entry.lastKey.assign((char*)(blockBuffer + bufferOffset), len);
         bufferOffset += len;
         memcpy(&entry.blockOffset, blockBuffer + bufferOffset, sizeof(u64));
         bufferOffset += sizeof(u64);
         index.push_back(move(entry));
      }
      currentOffset += SSTABLE_BLOCK_SIZE;  // go to next block
   }
}

struct SSTableIterator   // used to iterate through an sstable, entry by entry, loading a block at a time
{
   shared_ptr<SSTableReader> reader;
   u8* buffer;
   u64 currentBlockIndex =  0;
   u64 offsetInBlock = 0;
   bool cont = false;
   string key;
   string payload;
   bool isTombstone;

   SSTableIterator(shared_ptr<SSTableReader> reader) : reader(reader)
   {
      buffer = (u8*)aligned_alloc(4096, SSTABLE_BLOCK_SIZE);
      loadBlock(0);  // load first block of entries
   }
   ~SSTableIterator() { free(buffer); }

   void loadBlock(u64 blockIndex)  // load the next block of the sstable
   {
      if (blockIndex >= reader->getMetadata().numBlocks)
      {
         cont = false;
         return ;
      }
      u64 off = reader->getBlockOffset(blockIndex);
      if (pread(reader->getFd(), buffer, 4096, off) != 4096)  // read block
      {
         cont = false;
         return ;
      }
      bm.readCount.fetch_add(1);  // for rmb benchmark
      currentBlockIndex = blockIndex;
      offsetInBlock = 0;
      cont = true;
      next();  // go to the first entry in the block
   }
   bool next()
   {
      if (!cont) return false;
      if (offsetInBlock + sizeof(u16) > SSTABLE_BLOCK_SIZE)  // no space left in block even for the length of the key
      {
         loadBlock(currentBlockIndex + 1);
         return cont;
      }
      u16 keyLen;
      memcpy(&keyLen, buffer + offsetInBlock, sizeof(u16));
      u64 size = sizeof(u16) + keyLen + sizeof(u16);
      if (keyLen == 0 || offsetInBlock + size > SSTABLE_BLOCK_SIZE)  // check if safe to read
      {
         loadBlock(currentBlockIndex + 1);
         return cont;
      }
      u8* keyPtr = buffer + offsetInBlock + sizeof(u16);
      u16 payloadLen;
      memcpy(&payloadLen, keyPtr + keyLen, sizeof(u16));
      if (offsetInBlock + size + payloadLen + sizeof(u8) > SSTABLE_BLOCK_SIZE) // check if safe to read
      {
         loadBlock(currentBlockIndex + 1);
         return cont;
      }
      u8* payloadPtr = keyPtr + keyLen + sizeof(u16);
      key = string((char*)keyPtr, keyLen);         // read key, payload and tombstone
      payload = string((char*)payloadPtr, payloadLen);
      isTombstone = *(payloadPtr + payloadLen);
      offsetInBlock += payloadLen + keyLen + sizeof(u16) + sizeof(u16) + sizeof(u8); // move to next entry in block
      return true;
   }
};

struct MergeIteratorEntry   // a struct holding an iterator and a fileID so we can sort the iterators based on the id, so
{
   SSTableIterator* it;
   u64 fileID;
   bool operator>(const MergeIteratorEntry& other) const  // sort keys ascending
   {
      if (it->key != other.it->key) return it->key > other.it->key;
      return fileID < other.fileID;  // secondary sort id descending (higher id first)
   }
};


static once_flag lsmFlag;
struct LsmTree
{
private:
   static constexpr u64 LSM_ROOT_ID = 1;
   static constexpr size_t MEMTABLE_CAPACITY = 64 * 1024 * 1024;  // also approximately size of sstable
   Memtable memtable;
   mutex memtableMutex;
   atomic<size_t> memtableSize{0};
   vector<Memtable> immutableMemtables;
   mutex immutableMemtablesMutex;
   atomic<bool> levelCompacting[3] = {false, false, false};  // atomic flags to make sure only one compaction per level happens at a time
   thread flusherThread;
   condition_variable flusherCond;
   atomic<bool> shutDown{false};
   void backgroundFlush();  // writes sstables to L0 and triggers L0 compaction when necessary

   unordered_map<u64, shared_ptr<SSTableReader>> readerCache;  // we keep readers so we don't create new ones all the time
   shared_mutex readerMutex;
   shared_ptr<SSTableReader> getReader(u64 fileID);

   mutex updateMutexes[1024]; // for updatInPlace to lock the mutex of the hash of the key, we can't just mmake a mutex for each key as they would be a lot

   void flush();// flushing the memtable to immutable tables queue
   vector<SSTableMetadata> mergeOverlapping(const vector<u64>& filesToMerge, u16 targetLvl);
   void applyMerge(u32* count, Level0Entry* entries, const vector<u64>& toRemove, const vector<SSTableMetadata>& toAdd, u32 maxCap);
   void updateTables(u16 fromLvl, const vector<u64>& filesToCompact, u16 targetLvl, const vector<u64>& overlappingFiles, const vector<SSTableMetadata>& results);
   void compactL0();

public:
   unsigned slotId;
   atomic<bool> splitOrdered;
   PID rootPageId;

   LsmTree() : splitOrdered(false)
   {
      call_once(lsmFlag, []  // this only runs once, so the initialization is not done multiple times by different threads
      {
         GuardX<MetaDataPage> meta(0);
         if (meta->lsmRootPid == 0)
         {
            AllocGuard<LsmRootPage> root;
            memset(root.ptr, 0, pageSize);
            meta->lsmRootPid = root.pid;
            meta->dirty = true;
            root->nextSSTableId = 1;
            root->dirty = true;
         }
      });
      {
         GuardS<MetaDataPage> meta(0);
         this->rootPageId = meta->lsmRootPid;
         GuardX<LsmRootPage> root(this->rootPageId);
         if (root->nextSSTableId == 0)    // set counter to 1 if counter is not already stored on disk from a previous run
         {
            root->nextSSTableId = 1;
            root->dirty = true;
         }
      }
      flusherThread = std::thread(&LsmTree::backgroundFlush, this);  // start the background thread used for flushing to disk and compacting
   }
   ~LsmTree()
   {
      shutDown = true;
      flusherCond.notify_all();
      if (flusherThread.joinable()) flusherThread.join(); // wait for it to finish before shut down
   }
   void compactLevel(u16 sourceLevel);
   static u64 getFileId();
   void insert(span<u8> key, span<u8> payload);
   template<class Fn>
   bool lookup(span<u8> key, Fn fn);
   bool remove(span<u8> key);
   template<class Fn>
   void scanAsc(span<u8> key, Fn fn);
   template<class Fn>
   void scanDesc(span<u8> key, Fn fn)
   {
      return; // very inefficient so skip implementation, even worse than scanAsc
   }
   template<class Fn>
   bool updateInPlace(span<u8> key, Fn fn);
};
u64 LsmTree::getFileId()   // returns a new unique fileID and increments it
{
   GuardX<LsmRootPage> root(LSM_ROOT_ID);
   u64 id =  root->nextSSTableId++;
   root->dirty = true;
   return id;
}

shared_ptr<SSTableReader> LsmTree::getReader(u64 fileID)   // returns a reader for fileID (if there isn't one in cache it creates one)
{
   {
      shared_lock lock(readerMutex);   // faster if it is in cache
      auto it = readerCache.find(fileID);
      if (it != readerCache.end()) return it->second;
   }
   {
      unique_lock lock(readerMutex);
      auto it = readerCache.find(fileID);
      if (it != readerCache.end()) return it->second;  // check again
      auto reader = make_shared<SSTableReader>(fileID); // not in cache so create new one and also put it in cache
      readerCache[fileID] = reader;
      return reader;
   }

}

template <class Fn>
bool LsmTree::updateInPlace(span<u8> key, Fn fn)  // find the value and insert the updated value, no such thing as an update in place in an lsm tree
{     // we need to lock the specific entry while doing this, so this function is thread safe
      // we can't have a separate mutex for each key, but we can hash the key. we have 1024 buckets s oit is improbable they collide
   string_view view(reinterpret_cast<const char*>(key.data()),key.size());
   size_t bucket = hash<string_view>{}(view) % 1024;   // use c++ included hash
   lock_guard<mutex> updateLock(updateMutexes[bucket]);   // if already taken, thread will wait here

   u8 buffer[SSTABLE_BLOCK_SIZE];
   u16 payloadLen;
   bool found = false;
   this->lookup(key, [&](span<u8> payload)
   {
      if (payload.size() > SSTABLE_BLOCK_SIZE) return; // exists, but found stays false because payload is too big
      memcpy(buffer, payload.data(), payload.size());
      payloadLen = payload.size();
      found = true;  // this function doesn't run if we find it but is tombstone, if it is tombstone we don't want to update it anyway
   });
   if (!found) return false;
   fn(span<u8>(buffer, payloadLen));   // if found we apply the function and insert it
   this->insert(key, {buffer, payloadLen});   // TODO : payloadLen needs to be updated by fn
   return true;
}


SSTableMetadata SSTableWriter::writeMemtable(Memtable& memtable, u8 targetLevel) // convert a memtable to an sstable and writes it to L0
{
   assert(fileFD > 0);
   SSTableMetadata metadata = {};
   metadata.level = targetLevel;   // we return the metadata of the newly created sstable at the end
   metadata.fileID = this->fileID;
   memset(blockBuffer.data(), 0, SSTABLE_BLOCK_SIZE);
   for (auto& [key, entry] : memtable)  // we take every entry
   {
      if (firstKeyInTable.empty())
      {
         firstKeyInTable = key;
      }
      if (currentBlockSize + getRecordSize(key, entry) > SSTABLE_BLOCK_SIZE)   // check if entry fits in block
      {
         blockIndex.push_back({.lastKey = lastKeyInBlock, .blockOffset = blockOffset1});  // we store the offset and the alstKey for the blockIndex
         writeBlock(blockBuffer.data(), currentBlockSize);  // we write the block to disk and reset for the next block
         blockOffset1 = currentOffset;
         currentBlockSize = 0;
         memset(blockBuffer.data(), 0, SSTABLE_BLOCK_SIZE);
      }
      currentBlockSize += serializeRecord(blockBuffer.data(), currentBlockSize, key, entry);
      lastKeyInBlock = key;
   }
   if (currentBlockSize > 0) // write what is still in the buffer
   {
      blockIndex.push_back({.lastKey = lastKeyInBlock, .blockOffset = blockOffset1});
      writeBlock(blockBuffer.data(), currentBlockSize);
   }
   metadata.minKey = firstKeyInTable;  // finish metadata and write it at the start of the sstable
   metadata.maxKey = lastKeyInBlock;
   writeIndexAndMetadata(metadata);
   return metadata;
}

void SSTableWriter::addRecord(string_view key, string_view payload, bool isTombstone)   // just like writeMemtable but is used to add just one entry in an sstable
{
   if (firstKeyInTable.empty()) fill(blockBuffer.begin(), blockBuffer.end(), 0);
   u16 keyLen = key.size();
   u16 payloadLen = payload.size();
   u64 size = sizeof(u16) + keyLen + payloadLen + sizeof(u16) + sizeof(u8);
   if (currentBlockSize + size > SSTABLE_BLOCK_SIZE)  // check if entry will fit in buffer
   {
      blockIndex.push_back({lastKeyInBlock, blockOffset1});
      writeBlock(blockBuffer.data(), currentBlockSize);  // write buffer to disk and reset
      blockOffset1 = currentOffset;
      currentBlockSize = 0;
      fill(blockBuffer.begin(), blockBuffer.end(), 0);
   }
   u8* ptr = blockBuffer.data() + currentBlockSize; // currentBlockSize is also the offset in the block where we write
   memcpy(ptr, &keyLen, sizeof(u16)); ptr += sizeof(u16);   // put the entry in the buffer
   memcpy(ptr, key.data(), keyLen); ptr += keyLen;
   memcpy(ptr, &payloadLen, sizeof(u16)); ptr += sizeof(u16);
   memcpy(ptr, payload.data(), payloadLen); ptr += payloadLen;
   *ptr = (u8)isTombstone;
   currentBlockSize += size;
   lastKeyInBlock = string(key);   // update last key
   if (firstKeyInTable.empty()) firstKeyInTable = lastKeyInBlock;
}
void SSTableWriter::finish(SSTableMetadata& metadata)  // writes current block and then index and metadata
{
   if (currentBlockSize > 0)
   {
      blockIndex.push_back({lastKeyInBlock, blockOffset1});
      writeBlock(blockBuffer.data(), currentBlockSize);
   }
   metadata.minKey = firstKeyInTable;
   metadata.maxKey = lastKeyInBlock;
   writeIndexAndMetadata(metadata);
}


void LsmTree::insert(span<u8> key, span<u8> payload)  // adds an entry in the memtable
{
   if (key.size() + payload.size() > SSTABLE_BLOCK_SIZE - 32) // 32 bytes left empty for tombstone, key length, payload length and some extra
   {
      throw runtime_error("key / payload size too big");
   }
   for (u64 repeat = 0; ; repeat++) { // keep trying to insert until succesful
      try
         {
         string keyStr(reinterpret_cast<const char*>(key.data()), key.size());
         unique_lock<mutex> lock(memtableMutex);
         size_t entry_size;
         auto it = memtable.find(keyStr);  // check if entry key is already in the memtable map
         if (it != memtable.end())  // if it is the entry_size is the difference in payload sizes
         {
            entry_size = payload.size() - it->second.payload.size();
         }
         else   // if it is not in, it is total size of entry
         {
            entry_size = sizeof(MemtableEntry) + keyStr.size() + payload.size();
         }
         if (memtableSize.load() + entry_size > MEMTABLE_CAPACITY)  // if buffer full, flush
         {
            lock.unlock();
            flush();   // clear memtable, transfering it to an immtable and trigger l0 flush
            throw OLCRestartException();
         }
         if (it != memtable.end())
         {
            it->second = MemtableEntry(payload);
         }
         else
         {
            memtable.emplace(move(keyStr), MemtableEntry(payload));
         }
         memtableSize += entry_size;
         return; // Success, exit infinite loop
      }
      catch (const OLCRestartException&)  // retry if it was unsuccesful
      {
         yield(repeat);
      }
   }
}

void LsmTree::flush()
{
   Memtable newMemtable;
   {
      lock_guard<mutex> active_lock(memtableMutex);
      {
         lock_guard<mutex> immutable_lock(immutableMemtablesMutex);
         immutableMemtables.push_back(std::move(memtable));   // put the memtable in the immtable waiting queue, from where the flusher takes them and turns them into sstables
      }
      memtable = move(newMemtable);  // reset memtable
      memtableSize = 0;
   }
   flusherCond.notify_one();  // notify the flusher thread there is a new immtable to be flushed to L0

}

void LsmTree::backgroundFlush()
{
   workerThreadId = 100;  // unique id so it doesn't collide with other threads
   while (true)
   {
      Memtable memtableToFlush;
      {
         unique_lock<mutex> lock(immutableMemtablesMutex);
         flusherCond.wait(
            lock, [this]      // wait for signal that there is an immtable in the queue or that we are shutting down
         {
            return !immutableMemtables.empty() || shutDown;
         });
         if (shutDown && immutableMemtables.empty()) break;   // only stops when everything is flushed and shutDown has been signaled
         memtableToFlush = move(immutableMemtables.front());
         immutableMemtables.erase(immutableMemtables.begin()); // erase the memtable that we are about to flush from immtables
      }
      if (!memtableToFlush.empty())
      {
         try  // in case writer fails
         {
            u64 nextId = getFileId();
            SSTableMetadata metadata;
         {
            SSTableWriter writer(nextId);   // unique id for the sstable
            metadata = writer.writeMemtable(memtableToFlush, 0);   // we write the memtable to disk
         }  // writer goes out of scope here so the file is closed after this, otherwise it will stay open and we can't unlink it
            bool triggerCompaction = false;
            {
               GuardX<LsmRootPage> root(LSM_ROOT_ID);  // lock the root page where manifest is kept
               if (root->l0_count < LsmRootPage::MAX_L0_FILES)    // update the manifest of the entries of level 0
               {
                  u32 id = root->l0_count++;
                  root->l0Entries[id].fileID = metadata.fileID;
                  u64 minV = 0; u64 maxV = 0;
                  memcpy(&minV, metadata.minKey.data(), min<size_t>(metadata.minKey.size(), sizeof(u64)));
                  memcpy(&maxV, metadata.maxKey.data(), min<size_t>(metadata.maxKey.size(), sizeof(u64)));
                  root->l0Entries[id].minKey = __builtin_bswap64(minV);   // we store min and max key as u64 (just the first 8B of the key)
                  root->l0Entries[id].maxKey = __builtin_bswap64(maxV);   // so lookup doesn't need to open this file if key range doesn't match
                  root->dirty = true;                                     // we store them in big-endian for correct comparison
                  if (root->l0_count >= 5 && !levelCompacting[0].exchange(true))   // if a lot of files in l0 we trigger compaction
                  {                                             // we signal that level 0 is compacting so other threads dont compact at the same time
                     triggerCompaction = true;
                  }
               }
            }
               if (triggerCompaction)
               {
                  compactL0();
               }
         } catch (const exception& e)
         {
            cerr << "Failed flush" << e.what() << endl;
         }
      }
   }
}

void LsmTree::compactL0() {
   vector<u64> l0Files;   // files to be merged
   vector<u64> overlappingL1files;   // files to merge with from next level
   u16 mergeNum;
   {
      GuardS<LsmRootPage> root(LSM_ROOT_ID);
      mergeNum = min<u16>(root->l0_count, 4);   // batches of max 4 files
      if (mergeNum <= 0) {levelCompacting[0] = false;return;}
      u64 min, max;   // [min, max] will be the total key range of the batch]
      for (u16 i = 0; i < mergeNum; i++)
      {
         if (i == 0) {min = root->l0Entries[0].minKey; max = root->l0Entries[0].maxKey;}
         l0Files.push_back(root->l0Entries[i].fileID);
         if (root->l0Entries[i].minKey < min) min = root->l0Entries[i].minKey;
         if (root->l0Entries[i].maxKey > max) max = root->l0Entries[i].maxKey;
      }
      for (u16 i = 0; i < root->l1_count; i++)
      {
         u64 l1Min(root->l1Entries[i].minKey);
         u64 l1Max(root->l1Entries[i].maxKey);
         if (!(l1Max < min || l1Min > max))  // add only files from L1 that overlap with the files from L0
         {
            overlappingL1files.push_back(root->l1Entries[i].fileID);
         }
      }
   }
   vector<u64> filesToMerge = l0Files;
   filesToMerge.insert(filesToMerge.end(), overlappingL1files.begin(), overlappingL1files.end());   // add the files together in a single vector
   vector<SSTableMetadata> finishedFiles = mergeOverlapping(filesToMerge, 1); // merge all these files into the target level
   updateTables(0, l0Files, 1, overlappingL1files, finishedFiles);  // update the manifests

   {
      unique_lock lock(readerMutex);   // delete the readers of the merged files from the cache
      for (u64 oldID : filesToMerge)
      {
         readerCache.erase(oldID);
      }
   }

   for (u64 oldID : filesToMerge)   // delete the merged files from the disk
   {
      string path = "/tmp/sstables/" + to_string(oldID) + ".sst";
      struct stat st;
      if (stat(path.c_str(), &st) == 0)
      {
         bm.lsmSize -= st.st_size;  // used for space benchmark
      }
      if (unlink(path.c_str()) != 0)
      {
         if (errno != ENOENT)
         {
            perror("unlink failed");
         }
      }
   }
   levelCompacting[0] = false;
}


vector<SSTableMetadata> LsmTree::mergeOverlapping(const vector<u64>& filesToMerge, u16 targetLevel)
{
   priority_queue<MergeIteratorEntry, vector<MergeIteratorEntry>, greater<>> queue; // queue is sorted by key ascending based on the compare operator
   vector<unique_ptr<SSTableIterator>> iterators; // one iterator for each file that we are merging
   for (u64 fileID : filesToMerge)
   {
      try
      {
         auto reader = getReader(fileID);
         iterators.push_back(make_unique<SSTableIterator>(reader));
         if (iterators.back()->cont)
         {
            queue.push({iterators.back().get(), fileID});    // put the iterator in the queue
         }
      } catch (...)
      {
         continue;
      }
   }
   u64 newID;    // for the writer that will write the result of the merge
   unique_ptr<SSTableWriter> writer = nullptr;
   vector<SSTableMetadata> finishedFiles;  // vector to keep the metadatas of the resulting files (will need to update manifests)
   string lastKey = "";
   bool first = true;
   u64 mergeCount = 0;
   while (!queue.empty())  // merge until all iterators reach end
   {
      MergeIteratorEntry top = queue.top();   //we take the first iterator (it ahs the lowest key since they are sorted)
      queue.pop();   //take it out of the queue
      string currentKey(top.it->key);
      if (first || currentKey != lastKey)     // skip duplicates because the first one is most recent cause we have secondary sort fileID desc
      {
         mergeCount++;
         if (!writer)
         {
            newID = getFileId();
            writer = make_unique<SSTableWriter>(newID);
         }
         if (targetLevel != 3 || !top.it->isTombstone)   // delete tomstones in level 3
         {
            writer->addRecord(top.it->key, top.it->payload, top.it->isTombstone); // we don't use writeMemtable() because it would use too much RAM
         }                                                                        // to load all sstable in RAM as levels get bigger
         lastKey = move(currentKey);
         first = false;
         if (writer->getCurrentSize() > MEMTABLE_CAPACITY )    // if the sstable that we are writing noe gets too big we finish the sstable and start a new one
         {
            SSTableMetadata meta;   //write metadata
            meta.fileID = newID;
            meta.level = targetLevel;
            writer->finish(meta);  // add it to the beginning of the file
            finishedFiles.push_back(meta);

            writer.reset();  // reset writer so we have to create new one
         }
      }
      if (top.it->next())  // if not at the end of iterator
      {
         queue.push(top);  // we put it back in the queue so it has to sort it again
      }
   }
   if (writer)    // finish last sstable
   {
      SSTableMetadata meta;
      meta.fileID = newID;
      meta.level = targetLevel;
      writer->finish(meta);
      finishedFiles.push_back(meta);

      writer.reset();
   }
   return finishedFiles;
}
   // update the maifests for merged files
void LsmTree::updateTables(u16 fromLvl, const vector<u64>& filesToCompact, u16 targetLvl, const vector<u64>& overlappingFiles, const vector<SSTableMetadata>& results)
{
   vector<SSTableMetadata> remaining = results; // vector to keep the files we have to add if they don't all fit first try
   vector<u64> overlapping = overlappingFiles;  // these we delete from target level
   vector<u64> fromLvlVictims = filesToCompact;  // these we delete from the level that triggered the merge
   while (true)
   {
      bool needSpace = false;
      {
         GuardX<LsmRootPage> root(LSM_ROOT_ID); // exclusive lock manifests because we will modify them
         u32* targetCount;
         Level0Entry* targetEntries;   // we set these for the corresponding values
         u32 maxCapacity;

         if (targetLvl == 1)
         {
            targetCount = &root->l1_count;
            targetEntries = root->l1Entries;
            maxCapacity = LsmRootPage::MAX_L1_FILES;
         } else if (targetLvl == 2)
         {
            targetCount = &root->l2_count;
            targetEntries = root->l2Entries;
            maxCapacity = LsmRootPage::MAX_L2_FILES;
         } else if (targetLvl == 3) {
            targetCount = &root->l3_count;
            targetEntries = root->l3Entries;
            maxCapacity = LsmRootPage::MAX_L3_FILES;
            }
         else
         {
            throw runtime_error("target level not supported");
         }
         u32 survivingFiles = *targetCount - overlapping.size();
         u32 availableSlots = maxCapacity - survivingFiles;
         u32 toAddCount = min<u32>(availableSlots, (u32)remaining.size()); //we can add maximum how many spaces are available or all we have left to add
         if (availableSlots > 0 || remaining.empty())
         {
            if (!fromLvlVictims.empty())   // if merged files not deleted
            {
               if (fromLvl == 0)
               {
                  vector<Level0Entry> remainingL0;    // vector to keep those not to be deleted on level 0
                  for (u32 i = 0; i < root->l0_count; i++)
                  {
                     bool victim = false;
                     for (u64 id : fromLvlVictims)
                     {
                        if (root->l0Entries[i].fileID == id) {victim = true; break;}
                     }
                     if (!victim) remainingL0.push_back(root->l0Entries[i]);   // we keep those that are not on the victim list
                  }
                  root->l0_count = remainingL0.size();
                  for (u32 i = 0; i < root->l0_count; i++)
                  {
                     root->l0Entries[i] = remainingL0[i];   // put those that are not victims back on manifest in same order but shifted to the start
                  }
               }
               else
               {
                  u32* srcCount = fromLvl == 1 ? &root->l1_count : &root->l2_count;
                  Level0Entry* srcEntries = fromLvl == 1 ? root->l1Entries : root->l2Entries;
                  applyMerge(srcCount, srcEntries, fromLvlVictims, {}, (fromLvl == 1) ? LsmRootPage::MAX_L1_FILES : LsmRootPage::MAX_L2_FILES);
               }  // this adds nothingto the level that triggered the merge but deletes the victims
               fromLvlVictims.clear();  // no more need for this cause we deleted them
            }
            vector<SSTableMetadata> batch;  // to be added to target level
            for (u32 i = 0; i < toAddCount; i++)
            {
               batch.push_back(remaining.front()); // take from remaining and put in batch
               remaining.erase(remaining.begin());
            }
            applyMerge(targetCount, targetEntries, overlapping, batch, maxCapacity);  //batch will be added to target entries and overlapping will be deleted
            overlapping.clear();  //these were deleted so we clear it
            root->dirty = true; // mark root page dirty so buffer manager flushes it to disk
            if (remaining.empty())
            {
               return;  // done
            }
         }
         else{ needSpace = true;} // they did not all fit so we need to compact next level
      }
      if (needSpace)
      {
         compactLevel(targetLvl); // trigger next level compaction and let another thread work
      }
      this_thread::yield();
   }
}
   // updates manifest of level where we compacted from and of target level
void LsmTree::applyMerge(u32* count, Level0Entry* entries, const vector<u64>& toRemove, const vector<SSTableMetadata>& toAdd, u32 maxCap)
{
   vector<Level0Entry> keep; // entries not to be deleted
   for (u32 i = 0; i < *count; i++)
   {
      bool toDelete = false;
      for (u64 id : toRemove)
      {
         if (entries[i].fileID == id)
         {
            toDelete = true; break;   // delete those that have id in toRemove vector
         }
      }
      if (!toDelete)
      {
         keep.push_back(entries[i]);   // we keep the entry if it should not be deleted
      }
   }
   for (auto& m : toAdd)
   {
      Level0Entry entry;  // make a manifest entry
      entry.fileID = m.fileID;
      u64 minV = 0; u64 maxV = 0;  // set min and max key
      memcpy(&minV, m.minKey.data(), min<size_t>(m.minKey.size(), sizeof(u64)));
      memcpy(&maxV, m.maxKey.data(), min<size_t>(m.maxKey.size(), sizeof(u64)));
      entry.minKey = __builtin_bswap64(minV);   // swap bytes for big endian
      entry.maxKey = __builtin_bswap64(maxV);
      keep.push_back(entry);    // also keep it
   }
   sort(keep.begin(), keep.end(), [](const Level0Entry& a, const Level0Entry& b)
   {
      return a.minKey < b.minKey;  // sort them to be in ascending order by key
   });
   *count = min<u32>((u32)keep.size(), maxCap); // update the manifest count and entries
   for (u32 i = 0; i < *count; i++)
   {
      entries[i] = keep[i];
   }
}


// same as compactL0, but now we know ranges are not overlapping inside a level
void LsmTree::compactLevel(u16 sourceLevel)
{
   u16 destLevel = sourceLevel + 1;
   if (destLevel > 3 || sourceLevel <= 0) return;  // maximum L3
   if (levelCompacting[sourceLevel].exchange(true))  //if the level is already being compacted cancel this compaction
   {
      return;
   }
   vector<u64> sourceVictims;
   vector<u64> overlapping;
   {
      GuardS<LsmRootPage> root(LSM_ROOT_ID);    // lock root page and set source level and entries
      u32 count = sourceLevel == 1 ? root->l1_count : root->l2_count;
      Level0Entry* sourceEntries = sourceLevel == 1 ? root->l1Entries : root->l2Entries;
      if (count <= 0)
      {
         levelCompacting[sourceLevel] = false;  // cannot compact if level is empty so abort, reset compacting state
         return;
      }
      u32 batchSize = min<u32>(4, count); //compact a batch of maximum 4
      u64 min, max;  // for total key range of batch
      for (u16 i = 0; i < batchSize; i++)
      {
         if (i == 0) {min = sourceEntries[i].minKey; max = sourceEntries[i].maxKey;}
         sourceVictims.push_back(sourceEntries[i].fileID);
         if (sourceEntries[i].minKey < min) min = sourceEntries[i].minKey;  // find the smallest and biggest key
         if (sourceEntries[i].maxKey > max) max = sourceEntries[i].maxKey;
      }
      u32 destCount = destLevel == 2 ? root->l2_count : root->l3_count;  // select destination
      Level0Entry* destEntries = destLevel == 2 ? root->l2Entries : root->l3Entries;
      if (destEntries)
      {
         for (u32 i = 0; i < destCount; i++)
         {
            u64 minKey = destEntries[i].minKey;
            u64 maxKey = destEntries[i].maxKey;
            if (!(maxKey < min || minKey > max))
            {
               overlapping.push_back(destEntries[i].fileID);  //select the files from the destination level that overlap with the batch
            }
         }
      }
   }
   if (sourceVictims.empty()) return;
   vector<u64> filesToMerge =sourceVictims;   //put all files to be merged together
   filesToMerge.insert(filesToMerge.end(), overlapping.begin(), overlapping.end());
   vector<SSTableMetadata> results = mergeOverlapping(filesToMerge, destLevel);  // merge them into new sstables that we put on target level
   updateTables(sourceLevel, sourceVictims, destLevel, overlapping, results); // update manifests of target level and source level and trigger target level compaction in necessary
   {
      unique_lock lock(readerMutex);   // remove readers of deleted files from cache
      for (u64 oldID : filesToMerge)
      {
         readerCache.erase(oldID);
      }
   }
   for (u64 oldID : filesToMerge) // delete te files that were deleted during merging
   {
      string path = "/tmp/sstables/" + to_string(oldID) + ".sst";
      struct stat st;
      if (stat(path.c_str(), &st) == 0)
      {
         bm.lsmSize -= st.st_size;
      }
      if (unlink(path.c_str()) != 0)
      {
         if (errno != ENOENT) perror("unlink failed");;
      }
   }
   levelCompacting[sourceLevel] = false;  // done compacting so let this level be available for compaction again
}

template<class Fn>
bool LsmTree::lookup(span<u8> key, Fn fn)    // finds a key in the lsm tree and applies fn to it
{
   u64 keyInt = 0;
   memcpy(&keyInt, key.data(), min(key.size(), sizeof(u64)));
   keyInt = __builtin_bswap64(keyInt);   // for comparison swap the byte order because we store min and max keys as int, not as string
   string keyStr(reinterpret_cast<const char*>(key.data()), key.size());
   {                                                    // check memtable for key
      lock_guard<mutex> lock(memtableMutex);
      auto it = memtable.find(keyStr);
      if (it != memtable.end())   // if it is in memtable
      {
         if (!it->second.isTombstone)  // and is not deleted
         {
            span<u8> payload_span(reinterpret_cast<u8*>(it->second.payload.data()), it->second.payload.size());
            fn(payload_span);  // apply fn to the payload and return true
            return true;
         }
         return false;  // found tombstone so is deleted
      }
   }
   {
      lock_guard<mutex> lock(immutableMemtablesMutex);    // check immutable memtables from newest to oldest
      for (auto it = immutableMemtables.rbegin(); it != immutableMemtables.rend(); ++it)
      {
         auto mem_it = it->find(keyStr);    // they are also memtables so same here
         if (mem_it != it->end())
         {
            if (!mem_it->second.isTombstone)  // found key
            {
               span<u8> payload_span(reinterpret_cast<u8*>(mem_it->second.payload.data()), mem_it->second.payload.size());
               fn(payload_span);  // apply fn to it and return true
               return true;
            }
            return false; // found tombstone so the key was deleted
         }
      }
   }
   vector<u64> candidateFiles;   // now we search through sstables and select the ones that might have the key in L0
   {
      GuardS<LsmRootPage> root(LSM_ROOT_ID);
      for (int i = (int)root->l0_count - 1; i >= 0; i--)  // start from end (from most recent sstable)
      {
         u64 min = root->l0Entries[i].minKey;
         u64 max = root->l0Entries[i].maxKey;
         if (keyInt >= min && keyInt <= max)  // check if key is in the key range of the sstable
         {
            candidateFiles.push_back(root->l0Entries[i].fileID);  // if yes we save it so we know to check this sstable
         }
      }

      for (u64 fileID : candidateFiles)  // now we check the tables we saved
      {
         try
         {
            auto reader = getReader(fileID);
            u16 result = reader->lookup(key,fn);
            if (result == 1) return false; //found it but is tombstone, so we want to stop looking, as the first time we find it, it is the most recent entry
            if (result == 2) return true;
         } catch (...)
         {
            continue;
         }
      }
      auto checkLevel = [&](u32 count, Level0Entry* entries) -> u64   // function to check the files of a level, for levels 1,2,3
      {
         for (u32 i = 0; i < count; i++)   // in level 1,2,3 files don't have overlapping key ranges so the key can only be in one file
         {
            if (keyInt >= entries[i].minKey && keyInt <= entries[i].maxKey) return entries[i].fileID;   // we select the one file that can have the key
         }
         return 0;
      };
      u64 id;
      if ((id = checkLevel(root->l1_count, root->l1Entries))) if (getReader(id)->lookup(key, fn) == 2) return true;   // if it is tombstone we want to return false
      if ((id = checkLevel(root->l2_count, root->l2Entries))) if (getReader(id)->lookup(key, fn) == 2) return true;
      if ((id = checkLevel(root->l3_count, root->l3Entries))) if (getReader(id)->lookup(key, fn) == 2) return true;
   }
   return false;  // not found, finished all levels we can check
}

bool LsmTree::remove(span<u8> key)   // basically an insert with payload 0 and tombstone true
{
   span<u8> empty_payload = {};
   for (u64 repeat = 0; ; repeat++) {
      try
      {
         string keyStr(reinterpret_cast<const char*>(key.data()), key.size());  // cast key to string so we can use it
         unique_lock<mutex> lock(memtableMutex);  //lock memtable
         size_t entry_size;
         auto it = memtable.find(keyStr);  // check if entry key is in the memtable map
         if (it != memtable.end())  // if it is, the entry_size is the deleted payload size
         {
            entry_size = 0 - it->second.payload.size();
         }
         else   // if it is not in, the size is overhead of entry with 0 payload
         {
            entry_size = sizeof(MemtableEntry) + keyStr.size();
         }
         if (memtableSize.load() + entry_size > MEMTABLE_CAPACITY)  // if buffer full, flush
         {
            lock.unlock();  // release lock because flush needs to acces memtable
            flush();
            throw OLCRestartException();
         }
         if (it != memtable.end())  // if is in we update it
         {
            it->second = MemtableEntry(empty_payload, true);
         }
         else // if it is not already in memtable, we insert it
         {
            memtable.emplace(move(keyStr), MemtableEntry(empty_payload, true));
         }
         memtableSize += entry_size;
         return true; // Success, exit infinite loop
      }
      catch (const OLCRestartException&)  // was unsuccesfull (memtable was full) so retry
      {
         yield(repeat);
      }
   }
}

template <class Fn>
void LsmTree::scanAsc(span<u8> key, Fn fn)  // find the key and start reading the following keys in ascending order until fn returns false
{  // so after we find key, we then look for the next smallest key bigger then key in the entire tree, that means we have to merge all the sstables in a priority queue and iterate through it
   string startKey(reinterpret_cast<const char*>(key.data()), key.size());  // that can be done, but is extremely inefficient, so we will just search in the memtable
   lock_guard<mutex> lock(memtableMutex);
   auto memIt = memtable.lower_bound(startKey);  // returns iterator pointing to key or to next smallest entry if key not in
   /*u64 startKeyInt = 0;
   memcpy(&startKeyInt, key.data(), min(key.size(), sizeof(u64)));
   startKeyInt = __builtin_bswap64(startKeyInt);
   priority_queue<MergeIteratorEntry, vector<MergeIteratorEntry>, greater<>> queue;
   vector<unique_ptr<SSTableIterator>> iterators;
   auto memIt = memtable.lower_bound(startKey);
   vector<u64> candidates;
   {
      GuardS<LsmRootPage> root(LSM_ROOT_ID);
      auto addCandidates = [&](u32 count, Level0Entry* entries)
      {
         for (u32 i = 0; i < count; i++)
         {
            if (entries[i].maxKey >= startKeyInt)
            {
               candidates.push_back(entries[i].fileID);
            }
         }
      };
      addCandidates(root->l0_count, root->l0Entries);
      addCandidates(root->l1_count, root->l1Entries);
      addCandidates(root->l2_count, root->l2Entries);
      addCandidates(root->l3_count, root->l3Entries);

   }
   for (u64 id : candidates)
   {
      try
      {
         auto reader = getReader(id);
         auto it = make_unique<SSTableIterator>(reader);
         while (it->cont && it->key < startKey) {it->next();}  // go to start key
         if (it->cont)
         {
            queue.push({it.get(), id});
            iterators.push_back(move(it));
         }
      } catch (...) {}
   }
   string lastKey = "";
   */
   while (memIt != memtable.end())  //until we reach the end of the memtable
   {
      /*
      bool useMemtable = false;
      if (memIt != memtable.end())
      {
         if (queue.empty()) useMemtable = true;
         else if (memIt->first < queue.top().it->key) useMemtable = true;
      }
      string currentKey, currentPayload;
      bool isTombstone;
      if (useMemtable)
      {
         currentKey = memIt->first;
         currentPayload = memIt->second.payload;
         isTombstone = memIt->second.isTombstone;
         memIt++;
      } else
      {
         auto top = queue.top();
         queue.pop();
         currentKey = string(top.it->key);
         currentPayload = string(top.it->payload);
         isTombstone = top.it->isTombstone;
         if (top.it->next()) queue.push(top);
      }
      if (currentKey == lastKey) continue;
      lastKey = currentKey;
      if (isTombstone) continue;
      */
      if (memIt->second.isTombstone) {memIt++; continue;}  // we skip if it is tombstone

      bool cont = fn(span<u8>((u8*)memIt->first.data(), memIt->first.size()), span<u8>((u8*)memIt->second.payload.data(), memIt->second.payload.size()));
      if (!cont) break;   // if fn returned false we stop
      memIt++;  // go to next entry in memtable, since it is sorted
   }
}



static unsigned btreeslotcounter = 0;

BTree::BTree() : splitOrdered(false) {
   GuardX<MetaDataPage> page(metadataPageId);
   AllocGuard<BTreeNode> rootNode(true);
   slotId = btreeslotcounter++;
   page->roots[slotId] = rootNode.pid;
}

BTree::~BTree() {}

void BTree::trySplit(GuardX<BTreeNode>&& node, GuardX<BTreeNode>&& parent, span<u8> key, unsigned payloadLen)
{

   // create new root if necessary
   if (parent.pid == metadataPageId) {
      MetaDataPage* metaData = reinterpret_cast<MetaDataPage*>(parent.ptr);
      AllocGuard<BTreeNode> newRoot(false);
      newRoot->upperInnerNode = node.pid;
      metaData->roots[slotId] = newRoot.pid;
      parent = move(newRoot);
   }

   // split
   BTreeNode::SeparatorInfo sepInfo = node->findSeparator(splitOrdered.load());
   u8 sepKey[sepInfo.len];
   node->getSep(sepKey, sepInfo);

   if (parent->hasSpaceFor(sepInfo.len, sizeof(PID))) {  // is there enough space in the parent for the separator?
      node->splitNode(parent.ptr, sepInfo.slot, {sepKey, sepInfo.len});
      return;
   }

   // must split parent to make space for separator, restart from root to do this
   node.release();
   parent.release();
   ensureSpace(parent.ptr, {sepKey, sepInfo.len}, sizeof(PID));
}

void BTree::ensureSpace(BTreeNode* toSplit, span<u8> key, unsigned payloadLen)
{
   for (u64 repeatCounter=0; ; repeatCounter++) {
      try {
         GuardO<BTreeNode> parent(metadataPageId);
         GuardO<BTreeNode> node(reinterpret_cast<MetaDataPage*>(parent.ptr)->getRoot(slotId), parent);

         while (node->isInner() && (node.ptr != toSplit)) {
            parent = move(node);
            node = GuardO<BTreeNode>(parent->lookupInner(key), parent);
         }
         if (node.ptr == toSplit) {
            if (node->hasSpaceFor(key.size(), payloadLen))
               return; // someone else did split concurrently
            GuardX<BTreeNode> parentLocked(move(parent));
            GuardX<BTreeNode> nodeLocked(move(node));
            trySplit(move(nodeLocked), move(parentLocked), key, payloadLen);
         }
         return;
      } catch(const OLCRestartException&) { yield(repeatCounter); }
   }
}

void BTree::insert(span<u8> key, span<u8> payload)
{
   assert((key.size()+payload.size()) <= BTreeNode::maxKVSize);

   for (u64 repeatCounter=0; ; repeatCounter++) {
      try {
         GuardO<BTreeNode> parent(metadataPageId);
         GuardO<BTreeNode> node(reinterpret_cast<MetaDataPage*>(parent.ptr)->getRoot(slotId), parent);

         while (node->isInner()) {
            parent = move(node);
            node = GuardO<BTreeNode>(parent->lookupInner(key), parent);
         }

         if (node->hasSpaceFor(key.size(), payload.size())) {
            // only lock leaf
            GuardX<BTreeNode> nodeLocked(move(node));
            parent.release();
            nodeLocked->insertInPage(key, payload);
            return; // success
         }

         // lock parent and leaf
         GuardX<BTreeNode> parentLocked(move(parent));
         GuardX<BTreeNode> nodeLocked(move(node));
         trySplit(move(nodeLocked), move(parentLocked), key, payload.size());
         // insert hasn't happened, restart from root
      } catch(const OLCRestartException&) { yield(repeatCounter); }
   }
}

bool BTree::remove(span<u8> key)
{
   for (u64 repeatCounter=0; ; repeatCounter++) {
      try {
         GuardO<BTreeNode> parent(metadataPageId);
         GuardO<BTreeNode> node(reinterpret_cast<MetaDataPage*>(parent.ptr)->getRoot(slotId), parent);

         u16 pos;
         while (node->isInner()) {
            pos = node->lowerBound(key);
            PID nextPage = (pos == node->count) ? node->upperInnerNode : node->getChild(pos);
            parent = move(node);
            node = GuardO<BTreeNode>(nextPage, parent);
         }

         bool found;
         unsigned slotId = node->lowerBound(key, found);
         if (!found)
            return false;

         unsigned sizeEntry = node->slot[slotId].keyLen + node->slot[slotId].payloadLen;
         if ((node->freeSpaceAfterCompaction()+sizeEntry >= BTreeNodeHeader::underFullSize) && (parent.pid != metadataPageId) && (parent->count >= 2) && ((pos + 1) < parent->count)) {
            // underfull
            GuardX<BTreeNode> parentLocked(move(parent));
            GuardX<BTreeNode> nodeLocked(move(node));
            GuardX<BTreeNode> rightLocked(parentLocked->getChild(pos + 1));
            nodeLocked->removeSlot(slotId);
            if (rightLocked->freeSpaceAfterCompaction() >= (pageSize-BTreeNodeHeader::underFullSize)) {
               if (nodeLocked->mergeNodes(pos, parentLocked.ptr, rightLocked.ptr)) {
                  // XXX: should reuse page Id
               }
            }
         } else {
            GuardX<BTreeNode> nodeLocked(move(node));
            parent.release();
            nodeLocked->removeSlot(slotId);
         }
         return true;
      } catch(const OLCRestartException&) { yield(repeatCounter); }
   }
}

typedef u64 KeyType;

void handleSEGFAULT(int signo, siginfo_t* info, void* extra) {
   void* page = info->si_addr;
   if (bm.isValidPtr(page)) {
      cerr << "segfault restart " << bm.toPID(page) << endl;
      throw OLCRestartException();
   } else {
      cerr << "segfault " << page << endl;
      _exit(1);
   }
}

#ifdef USE_LSM_TREE    //choose at compile time if you want the lsm or btree version
   using Tree = LsmTree;
   #define TREE "LsmTree"
#else
   using Tree = BTree;
   #define TREE "BTree"
#endif


template <class Record>
struct vmcacheAdapter
{
   Tree tree;

   public:
   void scan(const typename Record::Key& key, const std::function<bool(const typename Record::Key&, const Record&)>& found_record_cb, std::function<void()> reset_if_scan_failed_cb) {
      u8 k[Record::maxFoldLength()];
      u16 l = Record::foldKey(k, key);
      u8 kk[Record::maxFoldLength()];
      tree.scanAsc({k, l}, [&](span<u8> key, span<u8> payload) {  //modified so lsm tree can also use them
         typename Record::Key typedKey;
         Record::unfoldKey(key.data(), typedKey);
         return found_record_cb(typedKey, *reinterpret_cast<const Record*>(payload.data()));
      });
   }
   // -------------------------------------------------------------------------------------
   void scanDesc(const typename Record::Key& key, const std::function<bool(const typename Record::Key&, const Record&)>& found_record_cb, std::function<void()> reset_if_scan_failed_cb) {
      u8 k[Record::maxFoldLength()];
      u16 l = Record::foldKey(k, key);
      u8 kk[Record::maxFoldLength()];
      tree.scanDesc({k, l}, [&](span<u8> keySpan, span<u8> payload) {   //modified so lsm tree can also use them
         typename Record::Key typedKey;
         Record::unfoldKey(keySpan.data(), typedKey);
         return found_record_cb(typedKey, *reinterpret_cast<const Record*>(payload.data()));
      });
   }
   // -------------------------------------------------------------------------------------
   void insert(const typename Record::Key& key, const Record& record) {
      u8 k[Record::maxFoldLength()];
      u16 l = Record::foldKey(k, key);
      tree.insert({k, l}, {(u8*)(&record), sizeof(Record)});
   }
   // -------------------------------------------------------------------------------------
   template<class Fn>
   void lookup1(const typename Record::Key& key, Fn fn) {
      u8 k[Record::maxFoldLength()];
      u16 l = Record::foldKey(k, key);
      bool succ = tree.lookup({k, l}, [&](span<u8> payload) {
         fn(*reinterpret_cast<const Record*>(payload.data()));
      });
      assert(succ);
   }
   // -------------------------------------------------------------------------------------
   template<class Fn>
   void update1(const typename Record::Key& key, Fn fn) {
      u8 k[Record::maxFoldLength()];
      u16 l = Record::foldKey(k, key);
      tree.updateInPlace({k, l}, [&](span<u8> payload) {
         fn(*reinterpret_cast<Record*>(payload.data()));
      });
   }
   // -------------------------------------------------------------------------------------
   // Returns false if the record was not found
   bool erase(const typename Record::Key& key) {
      u8 k[Record::maxFoldLength()];
      u16 l = Record::foldKey(k, key);
      return tree.remove({k, l});
   }
   // -------------------------------------------------------------------------------------
   template <class Field>
   Field lookupField(const typename Record::Key& key, Field Record::*f) {
      Field value;
      lookup1(key, [&](const Record& r) { value = r.*f; });
      return value;
   }

   u64 count() {
      u64 cnt = 0;
      tree.scanAsc({(u8*)nullptr, 0}, [&](BTreeNode& node, unsigned slot) { cnt++; return true; } );
      return cnt;
   }

   u64 countw(Integer w_id) {
      u8 k[sizeof(Integer)];
      fold(k, w_id);
      u64 cnt = 0;
      u8 kk[Record::maxFoldLength()];
      tree.scanAsc({k, sizeof(Integer)}, [&](BTreeNode& node, unsigned slot) {
         memcpy(kk, node.getPrefix(), node.prefixLen);
         memcpy(kk+node.prefixLen, node.getKey(slot), node.slot[slot].keyLen);
         if (memcmp(k, kk, sizeof(Integer))!=0)
            return false;
         cnt++;
         return true;
      });
      return cnt;
   }
};

template<class Fn>
void parallel_for(uint64_t begin, uint64_t end, uint64_t nthreads, Fn fn) {
   std::vector<std::thread> threads;
   uint64_t n = end-begin;
   if (n<nthreads)
      nthreads = n;
   uint64_t perThread = n/nthreads;
   for (unsigned i=0; i<nthreads; i++) {
     threads.emplace_back([&,i]() {
         uint64_t b = (perThread*i) + begin;
         uint64_t e = (i==(nthreads-1)) ? end : (b+perThread);
         fn(i, b, e);
      });
   }
   for (auto& t : threads)
      t.join();
}

void runAutomaticTests(LsmTree& lsm)
{
   cout << "LSM Correctness Tests" << endl;

   // Test 1: insert
   string key = "user123";
   string val = "hello_world";
   lsm.insert(span<u8>((u8*)key.data(), key.size()), span<u8>((u8*)val.data(), val.size()));

   bool found = lsm.lookup(span<u8>((u8*)key.data(), key.size()), [&](span<u8> payload) {
      assert(string((char*)payload.data(), payload.size()) == val);
   });
   assert(found);
   cout << "PASS basic insert and lookup" << endl;

   // Test 2: Update
   string val2 = "new_value";
   lsm.insert(span<u8>((u8*)key.data(), key.size()), span<u8>((u8*)val2.data(), val2.size()));
   lsm.lookup(span<u8>((u8*)key.data(), key.size()), [&](span<u8> payload) {
      assert(string((char*)payload.data(), payload.size()) == val2);
   });
   cout << "PASS insert new value with same key and lookup" << endl;

   // Test 3: Delete
   lsm.remove(span<u8>((u8*)key.data(), key.size()));
   bool stillExists = lsm.lookup(span<u8>((u8*)key.data(), key.size()), [&](span<u8> p){});
   assert(!stillExists);
   cout << "PASS Tombstone" << endl;

   // Test 4: 10 mil inserts to trigger flushes
   cout << "Testing large scale inserts (triggering flushes)..." << endl;
   for (int i = 0; i < 17000000; i++) {
      string k = "key_" + to_string(i);
      string v = "val_" + to_string(i);
      lsm.insert(span<u8>((u8*)k.data(), k.size()), span<u8>((u8*)v.data(), v.size()));
   }
   this_thread::sleep_for(chrono::seconds(20));
   // Verify a random key from the middle
   string searchK = "key_2700000";
   bool found1 = lsm.lookup(span<u8>((u8*)searchK.data(), searchK.size()), [&](span<u8> p){
       assert(string((char*)p.data(), p.size()) == "val_2700000");
   });
   assert(found1);
   cout << "PASS multi-level lookup" << endl;
}


int main(int argc, char** argv) {
   if (bm.useExmap) {
      struct sigaction action;
      action.sa_flags = SA_SIGINFO;
      action.sa_sigaction = handleSEGFAULT;
      if (sigaction(SIGSEGV, &action, NULL) == -1) {
         perror("sigusr: sigaction");
         exit(1);
      }
   }
   Tree bt;
   if (argc > 1 && string(argv[1]) == "--test")
   {
#ifdef USE_LSM_TREE
      runAutomaticTests(bt);
#endif
      return 0;
   }

   unsigned nthreads = envOr("THREADS", 1);
   u64 n = envOr("DATASIZE", 10);
   u64 runForSec = envOr("RUNFOR", 30);
   bool isRndread = envOr("RNDREAD", 0);

   u64 statDiff = 1e8;
   atomic<u64> txProgress(0);
   atomic<bool> keepRunning(true);
   auto systemName = bm.useExmap ? "exmap" : "vmcache";

   auto statFn = [&]() {
      cout << "ts,tx,rmb,wmb,system,threads,datasize,workload,batch" << endl;
      u64 cnt = 0;
      for (uint64_t i=0; i<runForSec; i++) {
         sleep(1);
         float rmb = (bm.readCount.exchange(0)*pageSize)/(1024.0*1024);
         float wmb = (bm.writeCount.exchange(0)*pageSize)/(1024.0*1024);
         u64 prog = txProgress.exchange(0);
         cout << cnt++ << "," << prog << "," << rmb << "," << wmb << "," << systemName << "," << nthreads << "," << n << "," << (isRndread?"rndread":"tpcc") << "," << bm.batch << endl;
      }
      keepRunning = false;
   };

   if (isRndread) {
      bt.splitOrdered = true;

      {
         // insert
         parallel_for(0, n, nthreads, [&](uint64_t worker, uint64_t begin, uint64_t end) {
            workerThreadId = worker;
            array<u8, 120> payload;
            for (u64 i=begin; i<end; i++) {
               union { u64 v1; u8 k1[sizeof(u64)]; };
               v1 = __builtin_bswap64(i);
               memcpy(payload.data(), k1, sizeof(u64));
               bt.insert({k1, sizeof(KeyType)}, payload);
            }
         });
      }
      cerr << "space: " << ((bm.allocCount.load()*pageSize) + bm.lsmSize.load())/(float)bm.gb << " GB " << endl;

      bm.readCount = 0;
      bm.writeCount = 0;
      thread statThread(statFn);

      parallel_for(0, nthreads, nthreads, [&](uint64_t worker, uint64_t begin, uint64_t end) {
         workerThreadId = worker;
         u64 cnt = 0;
         u64 start = rdtsc();
         while (keepRunning.load()) {
            union { u64 v1; u8 k1[sizeof(u64)]; };
            v1 = __builtin_bswap64(RandomGenerator::getRand<u64>(0, n));

            array<u8, 120> payload;
            bool succ = bt.lookup({k1, sizeof(u64)}, [&](span<u8> p) {
               memcpy(payload.data(), p.data(), p.size());
            });
            assert(succ);
            assert(memcmp(k1, payload.data(), sizeof(u64))==0);

            cnt++;
            u64 stop = rdtsc();
            if ((stop-start) > statDiff) {
               txProgress += cnt;
               start = stop;
               cnt = 0;
            }
         }
         txProgress += cnt;
      });

      statThread.join();
      return 0;
   }

   // TPC-C
   Integer warehouseCount = n;

   vmcacheAdapter<warehouse_t> warehouse;
   vmcacheAdapter<district_t> district;
   vmcacheAdapter<customer_t> customer;
   vmcacheAdapter<customer_wdl_t> customerwdl;
   vmcacheAdapter<history_t> history;
   vmcacheAdapter<neworder_t> neworder;
   vmcacheAdapter<order_t> order;
   vmcacheAdapter<order_wdc_t> order_wdc;
   vmcacheAdapter<orderline_t> orderline;
   vmcacheAdapter<item_t> item;
   vmcacheAdapter<stock_t> stock;

   TPCCWorkload<vmcacheAdapter> tpcc(warehouse, district, customer, customerwdl, history, neworder, order, order_wdc, orderline, item, stock, true, warehouseCount, true);

   {
      tpcc.loadItem();
      tpcc.loadWarehouse();

      parallel_for(1, warehouseCount+1, nthreads, [&](uint64_t worker, uint64_t begin, uint64_t end) {
         workerThreadId = worker;
         for (Integer w_id=begin; w_id<end; w_id++) {
            tpcc.loadStock(w_id);
            tpcc.loadDistrinct(w_id);
            for (Integer d_id = 1; d_id <= 10; d_id++) {
               tpcc.loadCustomer(w_id, d_id);
               tpcc.loadOrders(w_id, d_id);
            }
         }
      });
   }
   cerr << "space: " << ((bm.allocCount.load()*pageSize) + bm.lsmSize.load())/(float)bm.gb << " GB " << endl;
   bm.readCount = 0;
   bm.writeCount = 0;
   thread statThread(statFn);

   parallel_for(0, nthreads, nthreads, [&](uint64_t worker, uint64_t begin, uint64_t end) {
      workerThreadId = worker;
      u64 cnt = 0;
      u64 start = rdtsc();
      while (keepRunning.load()) {
         int w_id = tpcc.urand(1, warehouseCount); // wh crossing
         tpcc.tx(w_id);
         cnt++;
         u64 stop = rdtsc();
         if ((stop-start) > statDiff) {
            txProgress += cnt;
            start = stop;
            cnt = 0;
         }
      }
      txProgress += cnt;
   });

   statThread.join();
   cerr << "space: " << ((bm.allocCount.load()*pageSize) + bm.lsmSize.load())/(float)bm.gb << " GB " << endl;

   return 0;
}
