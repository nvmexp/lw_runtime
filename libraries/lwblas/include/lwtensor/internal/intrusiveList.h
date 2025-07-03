#pragma once

#include <assert.h>

namespace LWTENSOR_NAMESPACE
{

/**
 * \brief Intrusive list class
 * \details A list of objects that contain there link pointers, i.e. no addtl. memory required.
 * The main restriction is that an object can't be in more than one such list.
 * Element types need to inherit from the Member subclass.
 */
template<typename T>
class IntrusiveList
{

public:

    class Member
    {

        friend IntrusiveList;

    public:

        virtual ~Member() {}

    protected:

        void resetMember()
        {
            next_ = nullptr;
            prev_ = nullptr;
        }

    private:

        T* next_ = nullptr;
        T* prev_ = nullptr;
    };

    class Iterator
    {

        friend IntrusiveList;

    public:

        T* operator*()
        {
            return pos_;
        }

        bool operator!= (const Iterator& other) const
        {
            assert(list_ == other.list_);
            return pos_ != other.pos_;
        }

        void operator++(int)
        {
            pos_ = list_->getNext(pos_);
        }

    private:

        Iterator(const IntrusiveList* list_, T* pos_) : list_(list_), pos_(pos_) {}

        const IntrusiveList* list_;
        T* pos_;
    };

    friend Iterator;

    class ConstIterator
    {

        friend IntrusiveList;

    public:

        const T* operator*() const
        {
            return pos_;
        }

        bool operator!= (const ConstIterator& other) const
        {
            assert(list_ == other.list_);
            return pos_ != other.pos_;
        }

        void operator++(int)
        {
            pos_ = list_->getNext(pos_);
        }

    private:

        ConstIterator(const IntrusiveList* list_, const T* pos_) : list_(list_), pos_(pos_) {}

        const IntrusiveList* list_;
        const T* pos_;
    };
    friend ConstIterator;

    Iterator begin()
    {
        return {this, first_};
    }

    Iterator end()
    {
        return {this, nullptr};
    }

    ConstIterator cbegin() const
    {
        return {this, first_};
    }

    ConstIterator cend() const
    {
        return {this, nullptr};
    }

    Iterator erase(Iterator it)
    {
        T* next_ = getNext(it.pos_);
        erase(it.pos_);
        return Iterator{this, next_};
    }

    void pushBack(T* elem)
    {
        assert(elem != last_);
        setPrev(elem, last_);
        if (last_)
        {
            setNext(last_, elem);
        }
        last_ = elem;
        if (! first_)
        {
            first_ = elem;
        }
    }

    void pushFront(T* elem)
    {
        assert(elem != first_);
        setNext(elem, first_);
        if (first_)
        {
            setPrev(first_, elem);
        }
        first_ = elem;
        if (! last_)
        {
            last_ = elem;
        }
    }

    void popFront()
    {
        erase(first_);
    }

    void popBack()
    {
        erase(last_);
    }

    void erase(T* elem)
    {
        if (elem == last_)
        {
            last_ = getPrev(elem);
            if (last_)
            {
                setNext(last_, nullptr);
            }
        }
        else
        {
            setPrev(getNext(elem), getPrev(elem));
        }
        if (elem == first_)
        {
            first_ = getNext(elem);
            if (first_)
            {
                setPrev(first_, nullptr);
            }
        }
        else
        {
            setNext(getPrev(elem), getNext(elem));
        }
        setNext(elem, nullptr);
        setPrev(elem, nullptr);
    }

    void clear()
    {
        while (first_)
        {
            erase(first_);
        }
    }

    bool isEmpty()
    {
        return first_ == nullptr;
    }

    T* getFront()
    {
        return first_;
    }

    T* getBack()
    {
        return last_;
    }

    void moveToFront(T* elem)
    {
        if (first_ == elem)
        {
            return;
        }
        if (getPrev(elem) != nullptr || getNext(elem) != nullptr)
        {
            erase(elem);
        }
        pushFront(elem);
    }
 
private:

    Member* getMember(T* elem) const
    {
        return static_cast<Member*>(elem);
    }

    const Member* getMember(const T* elem) const
    {
        return static_cast<const Member*>(elem);
    }

    T* getNext(T* elem) const
    {
        return getMember(elem)->next_;
    }
    T* getPrev(T* elem) const
    {
        return getMember(elem)->prev_;
    }

    const T* getNext(const T* elem) const
    {
        return getMember(elem)->next_;
    }
    const T* getPrev(const T* elem) const
    {
        return getMember(elem)->prev_;
    }

    void setPrev(T* elem, T* to)
    {
        assert(elem != to);
        getMember(elem)->prev_ = to;
    }
    void setNext(T* elem, T* to)
    {
        assert(elem != to);
        getMember(elem)->next_ = to;
    }

    T* first_ = nullptr;
    T* last_ = nullptr;
};

}
