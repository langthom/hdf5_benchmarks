#pragma once

#include "configuration.h"

void C_single_dataset_write(Configuration const& config, double* measurementSecs);
void C_single_dataset_read(Configuration const& config, double* measurementsSecs);
