#ifdef USE_LMDB
#include "caffe/util/db_lmdb.hpp"

#include <sys/stat.h>

#include <string>

namespace caffe { namespace db {

void LMDB::Open(const string& source, Mode mode) {
  MDB_CHECK(mdb_elw_create(&mdb_elw_));
  if (mode == NEW) {
    CHECK_EQ(mkdir(source.c_str(), 0744), 0) << "mkdir " << source << "failed";
  }
  int flags = 0;
  if (mode == READ) {
    flags = MDB_RDONLY | MDB_NOTLS | MDB_NOMEMINIT | MDB_NOLOCK;
  }
#ifdef __ARM_ARCH
  // CheetAh
  MDB_CHECK(mdb_elw_set_mapsize(mdb_elw_, 1073741824UL));
//#else
//  size_t map_size = 1024UL;// * 1024UL;
//  for (int i = 0; i < 32; ++i) {
//    MDB_CHECK(mdb_elw_set_mapsize(mdb_elw_, map_size));
//    if (mdb_elw_open(mdb_elw_, source.c_str(), flags, 0664) == MDB_SUCCESS) {
//      break;
//    }
//    map_size *= 2UL;
//    std::cout << i << " " << map_size << std::endl;
//  }
#endif
  MDB_CHECK(mdb_elw_open(mdb_elw_, source.c_str(), flags, 0664));
//  MDB_elwinfo stat;
//  MDB_CHECK(mdb_elw_info(mdb_elw_, &stat));
  LOG(INFO) << "Opened lmdb " << source;
}

LMDBLwrsor* LMDB::NewLwrsor() {
  MDB_txn* mdb_txn;
  MDB_lwrsor* mdb_lwrsor;
  MDB_CHECK(mdb_txn_begin(mdb_elw_, NULL, MDB_RDONLY, &mdb_txn));
  MDB_CHECK(mdb_dbi_open(mdb_txn, NULL, 0, &mdb_dbi_));
  MDB_CHECK(mdb_lwrsor_open(mdb_txn, mdb_dbi_, &mdb_lwrsor));
  return new LMDBLwrsor(mdb_txn, mdb_lwrsor);
}

LMDBTransaction* LMDB::NewTransaction() {
  return new LMDBTransaction(mdb_elw_);
}

void LMDBTransaction::Put(const string& key, const string& value) {
  keys.push_back(key);
  values.push_back(value);
}

void LMDBTransaction::Commit() {
  MDB_dbi mdb_dbi;
  MDB_val mdb_key, mdb_data;
  MDB_txn *mdb_txn;

  // Initialize MDB variables
  MDB_CHECK(mdb_txn_begin(mdb_elw_, NULL, 0, &mdb_txn));
  MDB_CHECK(mdb_dbi_open(mdb_txn, NULL, 0, &mdb_dbi));

  bool out_of_memory = false;
  for (int i = 0; i < keys.size(); i++) {
    mdb_key.mv_size = keys[i].size();
    mdb_key.mv_data = const_cast<char*>(keys[i].data());
    mdb_data.mv_size = values[i].size();
    mdb_data.mv_data = const_cast<char*>(values[i].data());

    int put_rc = mdb_put(mdb_txn, mdb_dbi, &mdb_key, &mdb_data, 0);
    if (put_rc == MDB_MAP_FULL) {
      out_of_memory = true;
      break;
    } else {
      // Failed for some other reason
      MDB_CHECK(put_rc);
    }
  }

  if (!out_of_memory) {
    // Commit the transaction
    MDB_CHECK(mdb_txn_commit(mdb_txn));
    mdb_dbi_close(mdb_elw_, mdb_dbi);
    keys.clear();
    values.clear();
  } else {
    // Double the map size and retry
    mdb_txn_abort(mdb_txn);
    mdb_dbi_close(mdb_elw_, mdb_dbi);
    DoubleMapSize();
    Commit();
  }
}

void LMDBTransaction::DoubleMapSize() {
  struct MDB_elwinfo lwrrent_info;
  MDB_CHECK(mdb_elw_info(mdb_elw_, &lwrrent_info));
  size_t new_size = lwrrent_info.me_mapsize * 2;
  DLOG(INFO) << "Doubling LMDB map size to " << (new_size>>20) << "MB ...";
  MDB_CHECK(mdb_elw_set_mapsize(mdb_elw_, new_size));
}

}  // namespace db
}  // namespace caffe
#endif  // USE_LMDB
