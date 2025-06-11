#pragma once

#include <array>
#include <memory>
#include <string>

template<class... Args>
std::string formatedString(char const* format, Args&&... args) {
  auto size = std::snprintf(nullptr, 0, format, std::forward<Args>(args)...) + 1;
  auto buf = std::make_unique<char[]>(size);
  std::snprintf(buf.get(), size, format, std::forward<Args>(args)...);
  return std::string(buf.get(), buf.get() + size - 1);
}

struct Configuration {
  std::array<uint64_t, 3> tileDims;
  std::array<uint64_t, 3> numTiles;
  std::array<uint64_t, 3>* chunkSizes;
  bool random;
  int compressionLevel;
  std::string pathPrefix;

  inline std::string getFilename() const {
    auto s = this->globalShape();
    return formatedString("%sbench_%dx%dx%d_r%d_ch%d_c%d.h5", pathPrefix, s[0], s[1], s[2], random, chunkSizes == nullptr ? 0 : chunkSizes->at(0), compressionLevel);
  }

  inline std::string repr() const {
    return formatedString("Configuration(%dx%dx%d tiles of shape = %dx%dx%d; random: %s; chunking: %d)", numTiles[0], numTiles[1], numTiles[2], tileDims[0], tileDims[1], tileDims[2], random, chunkSizes == nullptr ? 0 : chunkSizes->at(0));
  }

  inline std::array<uint64_t, 3> globalShape() const {
    auto s = tileDims;
    s[0] *= numTiles[0];
    s[1] *= numTiles[1];
    s[2] *= numTiles[2];
    return s;
  }

  inline size_t cacheSize() const {
    constexpr size_t defaultCacheSize = 4 * 1024 * 1024;
    return chunkSizes ? (chunkSizes->at(0) * chunkSizes->at(1) * chunkSizes->at(2) * sizeof(float)) : defaultCacheSize;
  }
};

std::unique_ptr<float[]> GetData(Configuration const& config);

#define RETURN_ON_ERROR if (error == H5I_INVALID_HID) { return; }
