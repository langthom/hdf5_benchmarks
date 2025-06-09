#pragma once

#include "configuration.h"
#include <random>

std::unique_ptr<float[]> GetData(Configuration const& config) {
  uint64_t numTileElements = config.tileDims[0] * config.tileDims[1] * config.tileDims[2];
  std::unique_ptr<float[]> tile = std::make_unique<float[]>(numTileElements);

  if (config.constantPtr) {
    std::fill_n(tile.get(), numTileElements, *config.constantPtr);
  } else {
    std::random_device rd;
    std::mt19937 gen{rd()};
    std::uniform_real_distribution<float> dist{-5000, 5000};
    std::generate_n(tile.get(), numTileElements, [&dist, &gen] { return dist(gen); });
  }

  return tile;
}

