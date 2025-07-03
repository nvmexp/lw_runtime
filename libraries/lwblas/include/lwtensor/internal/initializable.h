#pragma once

#include <lwtensor/internal/defines.h>

namespace LWTENSOR_NAMESPACE
{
    /**
     * This class provides functionality corresponding to the initialization of an opaque
     * data structure.
     */
    template<uint32_t kInitializedMagicNumber>
    class Initializable
    {
        static const uint32_t kInitializedMagicNumber_ = kInitializedMagicNumber;
        static_assert(kInitializedMagicNumber_ != 0, "kInitializedMagicNumber invalid");

        public:
        Initializable() noexcept
        {
            this->unsetInitialized();
        }

        virtual ~Initializable(){}


        /**
         * \brief Mark this class as initialized.
         * \req None
         * \pre None
         * \returns None
         * \exception-guarantee nothrow
         * \behavior blocking, not reentrant, and thread safe
         */
        inline void unsetInitialized() noexcept
        {
            isInitialized_ = 0;
        }

        /**
         * \brief Modifies the classes' initialization status
         * \details By default sets it to initialized
         * \req None
         * \pre None
         * \returns None
         * \exception-guarantee nothrow
         * \behavior blocking, not reentrant, and thread safe
         */
        inline void setInitialized(bool initialized = true) noexcept
        {
            if (initialized)
            {
                isInitialized_ = kInitializedMagicNumber_;
            }
            else
            {
                unsetInitialized();
            }
        };

        /**
         * \brief Whether the this class was initialized.
         * \req None
         * \pre None
         * \returns Whether the class was initialized.
         * \retval true if the class was initialized
         * \retval false if the class was not initialized
         * \exception-guarantee nothrow
         * \behavior blocking, not reentrant, and thread safe
         */
        inline bool isInitialized() const noexcept
        {
            return (isInitialized_ == kInitializedMagicNumber_);
        };

        private:
        uint32_t isInitialized_ = 0;
    };
}
