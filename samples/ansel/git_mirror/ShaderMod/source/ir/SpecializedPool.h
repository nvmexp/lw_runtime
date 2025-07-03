#pragma once

namespace shadermod
{
namespace ir
{

    template <class T, size_t preallocCount = 20>
    class Pool
    {
    public:

        ~Pool()
        {
            destroy();
        }

        void preallocate()
        {
            allocateChunk();
        }

        void allocateChunk()
        {
            T * chunk = (T *)malloc(preallocCount * sizeof(T));
            
            // TODO[error]: add ptr check and throw 'out of mem' if needed
            
            m_allocatedChunks.push_back(chunk);

            m_freeElements.reserve(preallocCount);
            for (size_t i = 0, iend = preallocCount; i < iend; ++i)
            {
                m_freeElements.push_back(chunk+i);
            }
        }

        void destroy()
        {
            for (size_t i = 0, iend = m_allocatedChunks.size(); i < iend; ++i)
            {
                free(m_allocatedChunks[i]);
            }
            m_allocatedChunks.clear();
            m_freeElements.clear();
        }

        T * getElement()
        {
            if (m_freeElements.size() == 0)
                allocateChunk();

            T * element = m_freeElements.back();
            m_freeElements.pop_back();
            return element;
        }

        void putElement(T * element)
        {
            m_freeElements.push_back(element);
        }

        template<typename ... Types>
        T * newElement(Types&& ... constructorArgs)
        {
            T* el = getElement();

            try
            {
                new(el)T(std::forward<Types>(constructorArgs)...);
            }
            catch (...)
            {
                putElement(el);

                throw;
            }
            
            return el;
        }

        void deleteElement(T * element)
        {
            element->~T();
            putElement(element);
        }

    protected:

        std::vector <T *> m_freeElements;
        std::vector <T *> m_allocatedChunks;

    };

}
}
