vmcache_btree: vmcache.cpp tpcc/*pp
	g++ -DNDEBUG -O3 -std=c++20 -g -fnon-call-exceptions -fasynchronous-unwind-tables vmcache.cpp -o vmcache -laio
vmcache_lsm: vmcache.cpp tpcc/*pp
	g++  -O3 -std=c++20 -g -fnon-call-exceptions -fasynchronous-unwind-tables vmcache.cpp -DUSE_LSM_TREE -o vmcache -laio

clean:
	rm vmcache
