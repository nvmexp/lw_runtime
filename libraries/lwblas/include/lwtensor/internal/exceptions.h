#pragma once

#include <exception>
#include <string>

#include <lwtensor/internal/defines.h>

namespace LWTENSOR_NAMESPACE
{
   class IlwalidArgument : public std::exception
   {
      public:
         IlwalidArgument(const char * const desc, const int32_t argNb = -1) :
                 description(std::string("Invalid Argument")) {
             if(desc != nullptr)
             {
                 description += std::string(": ") + desc;
             }
             if(argNb != -1)
             {
                 description += std::to_string(argNb);
             }
         }

         const char * what() const throw()
         {
             return description.c_str();
         }
      private:
         std::string description;
   };
   class InternalError : public std::exception
   {
      public:
         InternalError(const char * const desc) : description(desc) {}

         const char * what() const throw()
         {
            return description.c_str();
         }
      private:
         std::string description;
   };
   class NotInitialized: public std::exception
   {
      public:
         const char * what() const throw()
         {
            return "Not Initialized.\n";
         }
      private:
   };
   class NotSupported: public std::exception
   {
      public:
         NotSupported() : description("") {}
         NotSupported(const char * const desc) : description(desc) {}
         const char * what() const throw()
         {
            return description.c_str();
         }
      private:
         std::string description;
   };
}
