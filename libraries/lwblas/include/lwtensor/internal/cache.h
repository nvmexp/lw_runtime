#pragma once

#include <mutex>
#include <unordered_map>
#include <cassert>
#include <iostream>
#include <fstream>
#include <exception>

#include <lwtensor.h>
#include <lwtensor/internal/initializable.h>
#include <lwtensor/internal/defines.h>
#include <lwtensor/internal/intrusiveList.h>

namespace LWTENSOR_NAMESPACE
{
    /**
     * Thread-safe software-managed cache for key-value pairs.
     */
    template<typename Key_, typename Value_>
    class Cache : public Initializable<241>
    {
        public:

            using Key = Key_;
            using Value = Value_;

            struct Metadata
            {
                void init()
                {
                    int deviceId;
                    lwdaGetDevice(&deviceId);
                    lwdaDeviceProp prop;
                    lwdaGetDeviceProperties(&prop, deviceId);

                    libraryVersion_ = lwtensorGetVersion();
                    lwdartVersion_ = lwtensorGetLwdartVersion();
                    major_ = prop.major;
                    minor_ = prop.minor;
                    multiProcessorCount_ = prop.multiProcessorCount;
                }

                bool operator==(const Metadata& other) const noexcept
                {
                    return lwdartVersion_ == other.lwdartVersion_ &&
                           libraryVersion_ == other.libraryVersion_ &&
                           major_ == other.major_ &&
                           minor_ == other.minor_ &&
                           multiProcessorCount_ == other.multiProcessorCount_;
                }

                bool operator!=(const Metadata& other) const noexcept
                {
                    return ! (*this == other);
                }

                size_t lwdartVersion_ = 0;
                size_t libraryVersion_ = 0;
                int32_t major_ = 0;
                int32_t minor_ = 0;
                int32_t multiProcessorCount_ = 0;
            };

            class Cacheline : public IntrusiveList<Cacheline>::Member
            {
                public:
                    enum Status
                    {
                        INVALID, ///< Indicates that the cacheline is invalid
                        VALID, ///< Indicates that the cacheline is valid
                    };

                    Cacheline() { reset(); }

                    Cacheline(const Key &key, const Value &value) : key_(key), value_(value), status_(VALID){}

                    void setStatus(const Status& status) { status_ = status; }
                    Status getStatus() const { return status_; }

                    void setKey(const Key& key) { key_ = key; }
                    Key getKey() const { return key_; }

                    void setValue(const Value& value) { value_ = value; }
                    Value getValue() const { return value_; }

                    uint32_t getAlgoCount() const { return algoCount_; }
                    void setAlgoCount(const uint32_t algoCount) { algoCount_ = algoCount; }
                    void incrementAlgoCount() noexcept { algoCount_++; }

                    void reset()
                    {
                        IntrusiveList<Cacheline>::Member::resetMember();
                        status_ = INVALID;
                        algoCount_ = 0;
                    }

                private:
                    Key key_;
                    Value value_;
                    Status status_;
                    uint32_t algoCount_; /// keeps track of how many kernels/algos have already been tried (only relevant w.r.t. incremental_autotuning)
            };

            using Hashmap = std::unordered_map<Key, Cacheline*>; /// maps key to index into cachelines_

            Cache() : numActiveCachelines_(0), cachelines_(nullptr), numCachelines_(0)
            {
                metadata_.init();
                unsetInitialized(); // as long as no cachelines are attached
            }

            virtual ~Cache()
            {
            }

            /**
             * Cache lookup. In the case of a cache hit (w.r.t. key) value will be
             * assigned the value stored in the cache; otherwise value will not be
             * modified.
             * \param[in] key Key to be searched for
             * \param[out] value This argument will updated in the case of a cache hit.
             * \param[out] algoCount Counter to keep track of the number of algorithms/kernels that have already been tried
             * \return true on hit, otherwise false
             * \behavior thread safe
             */
            bool get(const Key& key, Value &value, uint32_t &algoCount)
            {
                const std::lock_guard<Mutex> lock(mutex_);

                bool found = false;
                const auto it = lookup_.find(key);
                if( it != lookup_.end() )
                {
                    // cache hit
                    const auto cacheline = it->second;
                    moveCachelineToHead(cacheline);
                    value = cacheline->getValue();
                    algoCount = cacheline->getAlgoCount();
                    found = true;
                }

                return found;
            }

            bool get(const Key& key, Value &value)
            {
                uint32_t algoCount = 0;
                return this->get(key, value, algoCount);
            }

            /**
             * Inserts the provided key-value pair into the cache.
             *
             * Inserts the provided key-value pair into the cache; this also updates the
             * LRU.
             * \return true on successful insertion, false otherwise.
             * \behavior thread safe
             */
            bool put(const Key& key, const Value &value, const uint32_t algoCount = 0)
            {
                const std::lock_guard<Mutex> lock(mutex_);

                const auto it = lookup_.find(key);

                Cacheline* cacheline = nullptr;
                if (it != lookup_.end())
                {
                    cacheline = it->second;
                }
                if (cacheline == nullptr)
                {
                    cacheline = this->getFreeCacheline();
                }
                if (cacheline == nullptr)
                {
                    cacheline = this->evictCacheline();
                }
                assert(cacheline != nullptr);

                cacheline->setKey(key);
                cacheline->setValue(value);
                cacheline->setAlgoCount(algoCount);
                cacheline->setStatus(Cacheline::VALID);
                moveCachelineToHead(cacheline);
                lookup_.insert(std::pair<Key, Cacheline*>(key, cacheline));
                return true;
            }

            /**
             * Not thread-safe
             * \pre cachelines != nullptr and numCachelines > 0
             */
            lwtensorStatus_t attachCachelines(Cacheline* cachelines, uint32_t numCachelines)
            {
                cachelines_ = cachelines;
                numCachelines_ = numCachelines;
                for (unsigned int i = 0; i < numCachelines_; ++i)
                {
                    cachelines_[i].reset();
                }
                setInitialized();
                return LWTENSOR_STATUS_SUCCESS;
            }

            /**
             * Not thread-safe
             */
            lwtensorStatus_t detachCachelines()
            {
                if( cachelines_ == nullptr || numCachelines_ == 0 )
                {
                    return LWTENSOR_STATUS_NOT_SUPPORTED;
                }
                unsetInitialized();

                cachelines_ = nullptr;
                lruList_.clear();
                
                numCachelines_ = 0;
                return LWTENSOR_STATUS_SUCCESS;
            }

#ifdef DEBUG
            void print() const
            {
                printf("first: %p last: %p\n", lruList_.first_, lruList_.last_);
                for (auto it = lruList_.cbegin(); it != lruList_.cend(); it++)
                {
                    printf("%p: %p %p\n", *it, (*it)->prev_, (*it)->next_);
                }
            }
#endif

            lwtensorStatus_t writeToFile(const char filename[]) const
            {
                try
                {
                    std::ofstream file;
                    file.open(filename, std::ios::out | std::ios::binary);
                    if (! file.is_open() || ! file)
                    {
                        return LWTENSOR_STATUS_IO_ERROR;
                    }

                    /*
                     * Write meta-data information
                     */
                    file.write((const char*)&metadata_, sizeof(Metadata));
                    if (! file)
                    {
                        return LWTENSOR_STATUS_IO_ERROR;
                    }
                    file.write((const char*)&numActiveCachelines_, sizeof(numActiveCachelines_));
                    if (! file)
                    {
                        return LWTENSOR_STATUS_IO_ERROR;
                    }

                    uint32_t numCachelinesWritten = 0;
                    for (auto it = lruList_.cbegin(); it != lruList_.cend(); it++, numCachelinesWritten++)
                    {
                        file.write((const char*)*it, sizeof(Cacheline));
                        if (! file)
                        {
                            return LWTENSOR_STATUS_IO_ERROR;
                        }
                    }

                    if (numActiveCachelines_ != numCachelinesWritten)
                    {
                        return LWTENSOR_STATUS_INTERNAL_ERROR;
                    }

                    file.close();
                    if (! file)
                    {
                        return LWTENSOR_STATUS_IO_ERROR;
                    }
                    return LWTENSOR_STATUS_SUCCESS;
                }
                catch (std::exception& e)
                {
                    std::cerr << e.what() << std::endl;
                    return LWTENSOR_STATUS_IO_ERROR;
                }
            }

            lwtensorStatus_t readFromFile(const char filename[], uint32_t &numCachelinesRead)
            {
                try
                {
                    numCachelinesRead = 0;
                    std::ifstream file;
                    file.open(filename, std::ios::in | std::ios::binary);
                    if (! file.is_open() || ! file)
                    {
                        return LWTENSOR_STATUS_IO_ERROR;
                    }

                    /*
                     * Read meta-data information
                     */ 
                    Metadata metadata;

                    file.read((char*)&metadata, sizeof(Metadata));
                    if (! file)
                    {
                        return LWTENSOR_STATUS_IO_ERROR;
                    }

                    if (metadata_ != metadata)
                    {
                        return LWTENSOR_STATUS_ILWALID_VALUE;
                    }

                    // ensure that enough cachelines are available
                    decltype(numCachelines_) numCachelines = 0;

                    file.read((char*)&numCachelines, sizeof(numCachelines_));
                    if (! file)
                    {
                        return LWTENSOR_STATUS_IO_ERROR;
                    }

                    if (numCachelines > numCachelines_)
                    {
                        numCachelinesRead = numCachelines;
                        return LWTENSOR_STATUS_INSUFFICIENT_WORKSPACE;
                    }

                    file.read((char*)cachelines_, sizeof(Cacheline) * numCachelines);
                    if (! file)
                    {
                        return LWTENSOR_STATUS_IO_ERROR;
                    }

                    lruList_.clear();
                    lookup_.clear();
                    // update LRU and lookup table
                    for(int i=0; i < numCachelines; ++i)
                    {
                        lruList_.pushBack(&cachelines_[i]);
                        lookup_.insert(std::pair<Key, Cacheline*>(cachelines_[i].getKey(), &cachelines_[i]));
                    }
                    for(int i=numCachelines; i < numCachelines_; ++i)
                    {
                        cachelines_[i].reset();
                    }
                    numActiveCachelines_ = numCachelines;
                    numCachelinesRead = numCachelines;

                    return LWTENSOR_STATUS_SUCCESS;
                }
                catch (std::exception& e)
                {
                    std::cerr << e.what() << std::endl;
                    return LWTENSOR_STATUS_IO_ERROR;
                }
            }

            uint32_t size() const
            {
                return numActiveCachelines_;
            }

        private:

            /**
             * \return Returns a pointer to a free cacheline if a free cacheline is available, nullptr otherwise.
             */
            Cacheline* getFreeCacheline()
            {
                // The allocation is very naive at this point, since the current design
                // allows it to be (e.g., we are updating evicted cachelines); this
                // situation might change later on once we have invalid cachelines
                if( numActiveCachelines_ < numCachelines_ )
                {
                    auto cacheline = &cachelines_[numActiveCachelines_++];
                    cacheline->reset(); // return a clean cacheline
                    return cacheline;
                }
                else
                {
                    return nullptr;
                }
            }

            Cacheline* evictCacheline()
            {
                if (lruList_.isEmpty())
                {
                    return nullptr; // TODO throw? or keep it as an assert?
                }
                Cacheline *cacheline = lruList_.getBack(); // cacheline to be replaced
                lookup_.erase(cacheline->getKey());
                cacheline->setStatus(Cacheline::INVALID); // it's important not to reset() here since we still have to modify the pref_.{prev_,next_} and next_.{prev_,next_}
                return cacheline;
            }

            /**
             * This function emulates least-recently used behavior of the cache.
             * \warning Not thread-safe
             */
            void moveCachelineToHead(Cacheline *cacheline)
            {
//                printf("moveto head: front: %p back: %p, cl: %p, next: %p prev: %p\n",lruList_.first_,lruList_.last_,cacheline, cacheline->next_, cacheline->prev_);
                lruList_.moveToFront(cacheline);
            }

            uint32_t numActiveCachelines_;

            // Owned by the user:
            Cacheline *cachelines_; ///< pointer to first cacheline (it's assumed that this array contains at least numCachlines_ many cachelines)
            uint32_t numCachelines_; ///< number of cachelines available via cachelines_;
            Metadata metadata_;

            IntrusiveList<Cacheline> lruList_;

        protected:
            using Mutex = std::relwrsive_mutex;
            Hashmap lookup_; ///< lookup Key to index into cachelines_ (to enable O(1) accesses)
            mutable Mutex mutex_; ///< used for thread-safety
    };
}
