
#include "C_multiple_datasets.h"
#include "configuration.h"
#include <chrono>
#include <numeric>
#include <vector>
#include "hdf5.h"

/* Consider a file of the following structure (multiple datasets, one for each tile).

      HDF5 <filename> {
        GROUP "/" {
          GROUP "data" {
            DATASET "tile000" {
              DATATYPE H5T_NATIVE_FLOAT
              DATASPACE SIMPLE { (<config.tileDims> / <config.tileDims>) }
            }
            // all other tiles follow
          }
        }
      }

*/

void C_multiple_datasets_write(Configuration const& config, double* measurementSecs) {
  std::vector<double> tileSecs;
  std::string const targetH5File = config.getFilename();
  auto tileData = GetData(config);
  auto globalStart = std::chrono::high_resolution_clock::now();

  herr_t error;
  hid_t fileId  = H5Fcreate(targetH5File.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  hid_t groupId = H5Gcreate(fileId, "data", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  hid_t dcplId  = H5P_DEFAULT; // dcpl == dataset creation property list
  if (config.chunkSizes) {
    dcplId = H5Pcreate(H5P_DATASET_CREATE);
    error  = H5Pset_chunk(dcplId, 3, config.chunkSizes->data());
  }

  for (int tileZ = 0; tileZ < config.numTiles[0]; ++tileZ) {
    for (int tileY = 0; tileY < config.numTiles[1]; ++tileY) {
      for (int tileX = 0; tileX < config.numTiles[2]; ++tileX) {
        auto tileTimeBegin = std::chrono::high_resolution_clock::now();

        std::string const tileId = std::string("/data/tile") + std::to_string(tileZ) + std::to_string(tileY) + std::to_string(tileX);

        hid_t filespaceId = H5Screate_simple(3, config.tileDims.data(), NULL);
        hid_t datasetId   = H5Dcreate2(groupId, tileId.c_str(), H5T_NATIVE_FLOAT, filespaceId, H5P_DEFAULT, dcplId, H5P_DEFAULT);
        error = H5Dwrite(datasetId, H5T_NATIVE_FLOAT, H5S_ALL/*all memory in the tile dataset*/, filespaceId, H5P_DEFAULT, tileData.get());
        error = H5Dclose(datasetId);
        error = H5Sclose(filespaceId);

        auto tileTimeEnd = std::chrono::high_resolution_clock::now();
        tileSecs.push_back(std::chrono::duration<double>(tileTimeEnd - tileTimeBegin).count());
      }
    }
  }

  if (config.chunkSizes) error = H5Pclose(dcplId);
  error = H5Gclose(groupId);
  error = H5Fclose(fileId);
  RETURN_ON_ERROR;

  auto globalEnd = std::chrono::high_resolution_clock::now();
  measurementSecs[0] = std::chrono::duration<double>(globalEnd - globalStart).count();
  measurementSecs[1] = std::accumulate(tileSecs.cbegin(), tileSecs.cend(), 0.0) / tileSecs.size();
}


void C_multiple_datasets_read(Configuration const& config, double* measurementsSecs) {
  std::string const fname = config.getFilename();
  std::unique_ptr<float[]> readData;
  std::vector<double> readTileSecs, readRoiSecs, readXYSecs, readXZSecs, readYZSecs;

  herr_t error;
  hid_t fileId  = H5Fopen(fname.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  hid_t groupId = H5Gopen(fileId, "/data", H5P_DEFAULT);
  
  // Read full tiles
  readData.reset(new float[config.tileDims[0] * config.tileDims[1] * config.tileDims[2]]);

  for (int tileZ = 0; tileZ < config.numTiles[0]; ++tileZ) {
    for (int tileY = 0; tileY < config.numTiles[1]; ++tileY) {
      for (int tileX = 0; tileX < config.numTiles[2]; ++tileX) {
        auto tileReadBegin = std::chrono::high_resolution_clock::now();

        std::string const dataGroupPath = std::string("/data/tile") + std::to_string(tileZ) + std::to_string(tileY) + std::to_string(tileX);
        hid_t dataId = H5Dopen(groupId, dataGroupPath.c_str(), H5P_DEFAULT);
        error = H5Dread(dataId, H5T_IEEE_F32LE, H5S_ALL, H5S_ALL, H5P_DEFAULT, readData.get());
        error = H5Dclose(dataId);
        RETURN_ON_ERROR;

        auto tileReadEnd = std::chrono::high_resolution_clock::now();
        readTileSecs.push_back(std::chrono::duration<double>(tileReadEnd - tileReadBegin).count());
      }
    }
  }


  // Read a Roi of shape 128x128x128 crossing into 8 tiles.
  // TODO


  // Read each XY slice.
  {
    std::array<uint64_t, 3> origin{ 0, 0, 0 };
    std::array<uint64_t, 3> localOrigin = origin;
    std::array<uint64_t, 3> dim = config.globalShape();  dim[0] = 1;
    std::array<uint64_t, 3> localDim = config.tileDims;  localDim[0] = 1;
    readData.reset(new float[dim[1]*dim[2]]);
    auto localSliceData = std::make_unique<float[]>(localDim[1] * localDim[2]);

    hid_t localSliceMemorySpaceId = H5Screate_simple(3, localDim.data(), NULL);

    uint64_t numZ = config.globalShape()[0];
    for (uint64_t z = 0; z < numZ; ++z) {
      auto xyReadBegin = std::chrono::high_resolution_clock::now();

      auto zTile = z / config.tileDims[0];
      auto zInTile = z % config.tileDims[0];
      std::string const datasetPathPrefix = std::string("/data/tile") + std::to_string(zTile);
      localOrigin[0] = zInTile;

      for (int tileY = 0; tileY < config.numTiles[1]; ++tileY) {
        for (int tileX = 0; tileX < config.numTiles[2]; ++tileX) {
          std::string const datasetPath = datasetPathPrefix + std::to_string(tileY) + std::to_string(tileX);

          hid_t datasetId = H5Dopen2(groupId, datasetPath.c_str(), H5P_DEFAULT);
          hid_t fileSpaceId = H5Dget_space(datasetId);
          error = H5Sselect_hyperslab(fileSpaceId, H5S_SELECT_SET, localOrigin.data(), NULL, localDim.data(), NULL);
          error = H5Dread(datasetId, H5T_NATIVE_FLOAT, localSliceMemorySpaceId, fileSpaceId, H5P_DEFAULT, localSliceData.get());
          error = H5Dclose(datasetId);
          error = H5Sclose(fileSpaceId);

          // need to manually copy the locally read data into the correct position in the global slice.
          for (int row = 0; row < localDim[1]; ++row) {
            auto r = readData.get() + (tileY * localDim[1]*localDim[2]) + (tileX * localDim[2]);
            auto s = localSliceData.get() + row * localDim[2];
            std::memcpy(r, s, localDim[2] * sizeof(float));
          }
        }
      }

      auto xyReadEnd = std::chrono::high_resolution_clock::now();
      readXYSecs.push_back(std::chrono::duration<double>(xyReadEnd - xyReadBegin).count());
    }

    error = H5Sclose(localSliceMemorySpaceId);
  }


  // Read each XZ slice.
  {
    std::array<uint64_t, 3> origin{ 0, 0, 0 };
    std::array<uint64_t, 3> localOrigin = origin;
    std::array<uint64_t, 3> dim = config.globalShape();  dim[1] = 1;
    std::array<uint64_t, 3> localDim = config.tileDims;  localDim[1] = 1;
    readData.reset(new float[dim[0]*dim[2]]);
    auto localSliceData = std::make_unique<float[]>(localDim[0] * localDim[2]);

    hid_t localSliceMemorySpaceId = H5Screate_simple(3, localDim.data(), NULL);

    uint64_t numY = config.globalShape()[1];
    for (uint64_t y = 0; y < numY; ++y) {
      auto xzReadBegin = std::chrono::high_resolution_clock::now();

      auto yTile = y / config.tileDims[1];
      auto yInTile = y % config.tileDims[1];
      localOrigin[1] = yInTile;

      for (int tileZ = 0; tileZ < config.numTiles[0]; ++tileZ) {
        for (int tileX = 0; tileX < config.numTiles[2]; ++tileX) {
          std::string const datasetPath = std::string("/data/tile") + std::to_string(tileZ) + std::to_string(yTile) + std::to_string(tileX);

          hid_t datasetId = H5Dopen2(groupId, datasetPath.c_str(), H5P_DEFAULT);
          hid_t fileSpaceId = H5Dget_space(datasetId);
          error = H5Sselect_hyperslab(fileSpaceId, H5S_SELECT_SET, localOrigin.data(), NULL, localDim.data(), NULL);
          error = H5Dread(datasetId, H5T_NATIVE_FLOAT, localSliceMemorySpaceId, fileSpaceId, H5P_DEFAULT, localSliceData.get());
          error = H5Dclose(datasetId);
          error = H5Sclose(fileSpaceId);

          // need to manually copy the locally read data into the correct position in the global slice.
          for (int row = 0; row < localDim[0]; ++row) {
            auto r = readData.get() + (tileZ * localDim[0]*localDim[2]) + (tileX * localDim[2]);
            auto s = localSliceData.get() + row * localDim[2];
            std::memcpy(r, s, localDim[2] * sizeof(float));
          }
        }
      }

      auto xzReadEnd = std::chrono::high_resolution_clock::now();
      readXZSecs.push_back(std::chrono::duration<double>(xzReadEnd - xzReadBegin).count());
    }

    error = H5Sclose(localSliceMemorySpaceId);
  }

  // Read each YZ slice.
  {
    std::array<uint64_t, 3> origin{ 0, 0, 0 };
    std::array<uint64_t, 3> localOrigin = origin;
    std::array<uint64_t, 3> dim = config.globalShape();  dim[2] = 1;
    std::array<uint64_t, 3> localDim = config.tileDims;  localDim[2] = 1;
    readData.reset(new float[dim[0]*dim[1]]);
    auto localSliceData = std::make_unique<float[]>(localDim[0] * localDim[1]);

    hid_t localSliceMemorySpaceId = H5Screate_simple(3, localDim.data(), NULL);

    uint64_t numX = config.globalShape()[1];
    for (uint64_t x = 0; x < numX; ++x) {
      auto yzReadBegin = std::chrono::high_resolution_clock::now();

      auto xTile = x / config.tileDims[1];
      auto xInTile = x % config.tileDims[1];
      localOrigin[2] = xInTile;

      for (int tileZ = 0; tileZ < config.numTiles[0]; ++tileZ) {
        for (int tileY = 0; tileY < config.numTiles[1]; ++tileY) {
          std::string const datasetPath = std::string("/data/tile") + std::to_string(tileZ) + std::to_string(tileY) + std::to_string(xTile);

          hid_t datasetId = H5Dopen2(groupId, datasetPath.c_str(), H5P_DEFAULT);
          hid_t fileSpaceId = H5Dget_space(datasetId);
          error = H5Sselect_hyperslab(fileSpaceId, H5S_SELECT_SET, localOrigin.data(), NULL, localDim.data(), NULL);
          error = H5Dread(datasetId, H5T_NATIVE_FLOAT, localSliceMemorySpaceId, fileSpaceId, H5P_DEFAULT, localSliceData.get());
          error = H5Dclose(datasetId);
          error = H5Sclose(fileSpaceId);

          // need to manually copy the locally read data into the correct position in the global slice.
          for (int row = 0; row < localDim[0]; ++row) {
            auto r = readData.get() + (tileZ * localDim[0]*localDim[1]) + (tileY * localDim[1]);
            auto s = localSliceData.get() + row * localDim[1];
            std::memcpy(r, s, localDim[1] * sizeof(float));
          }
        }
      }

      auto yzReadEnd = std::chrono::high_resolution_clock::now();
      readYZSecs.push_back(std::chrono::duration<double>(yzReadEnd - yzReadBegin).count());
    }

    error = H5Sclose(localSliceMemorySpaceId);
  }

  error = H5Gclose(groupId);
  error = H5Fclose(fileId);
  RETURN_ON_ERROR;


  auto mean = [](std::vector<double> const& measurements) {
    return std::accumulate(measurements.cbegin(), measurements.cend(), 0.0) / measurements.size();
  };

  measurementsSecs[0] = mean(readTileSecs);
  measurementsSecs[1] = mean(readRoiSecs);
  measurementsSecs[2] = mean(readXYSecs);
  measurementsSecs[3] = mean(readXZSecs);
  measurementsSecs[4] = mean(readYZSecs);
}

