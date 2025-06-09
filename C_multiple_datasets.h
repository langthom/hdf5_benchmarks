#pragma once

#include "configuration.h"

void C_multiple_datasets_write(Configuration const& config, double* measurementSecs);
void C_multiple_datasets_read(Configuration const& config, double* measurementsSecs);
