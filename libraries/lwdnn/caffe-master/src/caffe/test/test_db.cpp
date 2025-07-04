#if defined(USE_LEVELDB) && defined(USE_LMDB)
#include <memory>
#include <string>

#include "gtest/gtest.h"

#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/io.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

using std::unique_ptr;

template <typename TypeParam>
class DBTest : public ::testing::Test {
 protected:
  DBTest()
      : backend_(TypeParam::backend),
      root_images_(string(EXAMPLES_SOURCE_DIR) + string("images/")) {}

  virtual void SetUp() {
    source_ = MakeTempDir();
    source_ += "/db";
    string keys[] = {"cat.jpg", "fish-bike.jpg"};
    LOG(INFO) << "Using temporary db " << source_;
    unique_ptr<db::DB> db(db::GetDB(TypeParam::backend));
    db->Open(this->source_, db::NEW);
    unique_ptr<db::Transaction> txn(db->NewTransaction());
    for (int i = 0; i < 2; ++i) {
      Datum datum;
      ReadImageToDatum(root_images_ + keys[i], i, &datum);
      string out;
      CHECK(datum.SerializeToString(&out));
      txn->Put(keys[i], out);
    }
    txn->Commit();
  }

  virtual ~DBTest() { }

  DataParameter_DB backend_;
  string source_;
  string root_images_;
};

struct TypeLevelDB {
  static DataParameter_DB backend;
};
DataParameter_DB TypeLevelDB::backend = DataParameter_DB_LEVELDB;

struct TypeLMDB {
  static DataParameter_DB backend;
};
DataParameter_DB TypeLMDB::backend = DataParameter_DB_LMDB;

// typedef ::testing::Types<TypeLmdb> TestTypes;
typedef ::testing::Types<TypeLevelDB, TypeLMDB> TestTypes;

TYPED_TEST_CASE(DBTest, TestTypes);

TYPED_TEST(DBTest, TestGetDB) {
  unique_ptr<db::DB> db(db::GetDB(TypeParam::backend));
}

TYPED_TEST(DBTest, TestNext) {
  unique_ptr<db::DB> db(db::GetDB(TypeParam::backend));
  db->Open(this->source_, db::READ);
  unique_ptr<db::Cursor> cursor(db->NewLwrsor());
  EXPECT_TRUE(cursor->valid());
  cursor->Next();
  EXPECT_TRUE(cursor->valid());
  cursor->Next();
  EXPECT_FALSE(cursor->valid());
}

TYPED_TEST(DBTest, TestSeekToFirst) {
  unique_ptr<db::DB> db(db::GetDB(TypeParam::backend));
  db->Open(this->source_, db::READ);
  unique_ptr<db::Cursor> cursor(db->NewLwrsor());
  cursor->Next();
  cursor->SeekToFirst();
  EXPECT_TRUE(cursor->valid());
  string key = cursor->key();
  Datum datum;
  EXPECT_TRUE(cursor->parse(&datum));
  EXPECT_EQ(key, "cat.jpg");
  EXPECT_EQ(datum.channels(), 3);
  EXPECT_EQ(datum.height(), 360);
  EXPECT_EQ(datum.width(), 480);
}

TYPED_TEST(DBTest, TestKeyValue) {
  unique_ptr<db::DB> db(db::GetDB(TypeParam::backend));
  db->Open(this->source_, db::READ);
  unique_ptr<db::Cursor> cursor(db->NewLwrsor());
  EXPECT_TRUE(cursor->valid());
  string key = cursor->key();
  Datum datum;
  EXPECT_TRUE(cursor->parse(&datum));
  EXPECT_EQ(key, "cat.jpg");
  EXPECT_EQ(datum.channels(), 3);
  EXPECT_EQ(datum.height(), 360);
  EXPECT_EQ(datum.width(), 480);
  cursor->Next();
  EXPECT_TRUE(cursor->valid());
  key = cursor->key();
  EXPECT_TRUE(cursor->parse(&datum));
  EXPECT_EQ(key, "fish-bike.jpg");
  EXPECT_EQ(datum.channels(), 3);
  EXPECT_EQ(datum.height(), 323);
  EXPECT_EQ(datum.width(), 481);
  cursor->Next();
  EXPECT_FALSE(cursor->valid());
}

TYPED_TEST(DBTest, TestWrite) {
  unique_ptr<db::DB> db(db::GetDB(TypeParam::backend));
  db->Open(this->source_, db::WRITE);
  unique_ptr<db::Transaction> txn(db->NewTransaction());
  Datum datum;
  ReadFileToDatum(this->root_images_ + "cat.jpg", 0, &datum);
  string out;
  CHECK(datum.SerializeToString(&out));
  txn->Put("cat.jpg", out);
  ReadFileToDatum(this->root_images_ + "fish-bike.jpg", 1, &datum);
  CHECK(datum.SerializeToString(&out));
  txn->Put("fish-bike.jpg", out);
  txn->Commit();
}

}  // namespace caffe

#endif  // USE_LEVELDB, USE_LMDB
