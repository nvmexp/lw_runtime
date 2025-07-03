#pragma once

class CCmd
{
public:
    CCmd(void);
    ~CCmd(void);

    virtual void Capture() = 0;
};
