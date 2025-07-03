#!/usr/bin/elw bash
# set default environment variables

set -e

WITH_CMAKE=${WITH_CMAKE:-false}
WITH_PYTHON3=${WITH_PYTHON3:-false}
WITH_LWDA=${WITH_LWDA:-true}
WITH_LWDNN=${WITH_LWDNN:-false}
