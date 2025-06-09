#pragma once

#include <array>
#include <memory>
#include <string>

template<class... Args>
std::string formatedString(char const* format, Args&&... args) {
  auto size = std::snprintf(nullptr, 0, format, std::forward<Args>(args)...);
  std::string str(size + 1, '\0');
  std::sprintf(&str[0], format, std::forward<Args>(args)...);
  return str;
}

struct Configuration {
  std::array<uint64_t, 3> tileDims;
  int numTiles;
  float* constantPtr;
  std::array<uint64_t, 3>* chunkSizes;

  std::string getFilename() const {
    return formatedString("D:/bench_%dx%dx%d_n%d_r%d_c%d.h5", tileDims[0], tileDims[1], tileDims[2], numTiles, (constantPtr == nullptr), chunkSizes == nullptr ? 0 : chunkSizes->at(0));
  }

  std::string repr() const {
    return formatedString("Configuration(%d tiles of shape = %dx%dx%d; random: %s; chunking: %d)", numTiles, tileDims[0], tileDims[1], tileDims[2], (constantPtr == nullptr), chunkSizes == nullptr ? 0 : chunkSizes->at(0));
  }
};

std::unique_ptr<float[]> GetData(Configuration const& config);

#define RETURN_ON_ERROR if (error == H5I_INVALID_HID) { return; }
