#ifndef CAFFE_UTIL_DB_HPP
#define CAFFE_UTIL_DB_HPP

#include <string>

#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe { namespace db {

enum Mode { READ, WRITE, NEW };

class Cursor {
 public:
  Cursor() { }
  virtual ~Cursor() { }
  virtual void SeekToFirst() = 0;
  virtual void Next() = 0;
  virtual string key() const = 0;
  virtual string value() const = 0;
  virtual const void* data() const = 0;
  virtual size_t size() const = 0;
  virtual bool parse(Datum* datum) const = 0;
  virtual bool parse(AnnotatedDatum* datum) const = 0;
  virtual bool parse(C2TensorProtos* c2p) const = 0;
  virtual bool valid() const = 0;

  DISABLE_COPY_MOVE_AND_ASSIGN(Cursor);
};

class Transaction {
 public:
  Transaction() { }
  virtual ~Transaction() { }
  virtual void Put(const string& key, const string& value) = 0;
  virtual void Commit() = 0;

  DISABLE_COPY_MOVE_AND_ASSIGN(Transaction);
};

class DB {
 public:
  DB() { }
  virtual ~DB() { }
  virtual void Open(const string& source, Mode mode) = 0;
  virtual void Close() = 0;
  virtual Cursor* NewLwrsor() = 0;
  virtual Transaction* NewTransaction() = 0;

  DISABLE_COPY_MOVE_AND_ASSIGN(DB);
};

DB* GetDB(DataParameter::DB backend);
DB* GetDB(const string& backend);

}  // namespace db
}  // namespace caffe

#endif  // CAFFE_UTIL_DB_HPP
