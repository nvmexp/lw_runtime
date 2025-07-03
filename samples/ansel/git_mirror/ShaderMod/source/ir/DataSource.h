#pragma once

namespace shadermod
{
namespace ir
{

    class DataSource
    {
    public:

        enum class DataType
        {
            kPass,
            kTexture,

            kNUM_ENTRIES
        };

        virtual DataType getDataType() const = 0;

    protected:
    };

}
}
