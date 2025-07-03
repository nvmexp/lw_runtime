/*
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */


#pragma once

#include <thrust/system/lwca/error.h>
#include <thrust/system/lwca/detail/guarded_lwca_runtime_api.h>

namespace thrust
{

namespace system
{


error_code make_error_code(lwca::errc::errc_t e)
{
  return error_code(static_cast<int>(e), lwca_category());
} // end make_error_code()


error_condition make_error_condition(lwca::errc::errc_t e)
{
  return error_condition(static_cast<int>(e), lwca_category());
} // end make_error_condition()


namespace lwca_cub
{

namespace detail
{


class lwca_error_category
  : public error_category
{
  public:
    inline lwca_error_category(void) {}

    inline virtual const char *name(void) const
    {
      return "lwca";
    }

    inline virtual std::string message(int ev) const
    {
      char const* const unknown_str  = "unknown error";
      char const* const unknown_name = "lwcaErrorUnknown";
      char const* c_str  = ::lwcaGetErrorString(static_cast<lwcaError_t>(ev));
      char const* c_name = ::lwcaGetErrorName(static_cast<lwcaError_t>(ev));
      return std::string(c_name ? c_name : unknown_name)
           + ": " + (c_str ? c_str : unknown_str);
    }

    inline virtual error_condition default_error_condition(int ev) const
    {
      using namespace lwca::errc;

      if(ev < ::lwcaErrorApiFailureBase)
      {
        return make_error_condition(static_cast<errc_t>(ev));
      }

      return system_category().default_error_condition(ev);
    }
}; // end lwca_error_category

} // end detail

} // end namespace lwca_cub


const error_category &lwca_category(void)
{
  static const thrust::system::lwca_cub::detail::lwca_error_category result;
  return result;
}


} // end namespace system

} // end namespace thrust

