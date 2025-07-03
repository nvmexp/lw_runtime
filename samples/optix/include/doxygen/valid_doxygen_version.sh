#!/bin/bash

if [ $1 == '1.8.4' ] || [ $1 == '1.8.5' ] || [ $1 == '1.8.6' ]
then
    echo true
else
    echo OptiX API documentation error: The Doxygen version should be 1.8.4, 1.8.5, or 1.8.6, not $1
fi
