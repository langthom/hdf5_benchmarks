
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
            DATASET "tile0" {
              DATATYPE H5T_IEEE_F32LE
              DATASPACE SIMPLE { (<config.tileDims> / <config.tileDims>) }
            }
            // all other tiles follow
          }
        }
      }

*/

void C_multiple_datasets_write(Configuration const& config, double* measurementSecs) {
  std::vector<double> tileSecs;
  tileSecs.reserve(config.numTiles);

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

  for (int tileId = 0; tileId < config.numTiles; ++tileId) {
    auto dataGroupBegin = std::chrono::high_resolution_clock::now();

    std::string const dataGroupPath = std::string("/data/tile") + std::to_string(tileId);

    hid_t dataSpaceId = H5Screate_simple(3, config.tileDims.data(), NULL);
    hid_t datasetId = H5Dcreate(groupId, dataGroupPath.c_str(), H5T_NATIVE_FLOAT, dataSpaceId, H5P_DEFAULT, dcplId, H5P_DEFAULT);

    error = H5Dwrite(datasetId, H5T_NATIVE_FLOAT, H5S_ALL, dataSpaceId, H5P_DEFAULT, tileData.get());
    RETURN_ON_ERROR;
    error = H5Dclose(datasetId);
    RETURN_ON_ERROR;
    error = H5Sclose(dataSpaceId);
    RETURN_ON_ERROR;

    auto dataGroupEnd = std::chrono::high_resolution_clock::now();
    tileSecs.push_back(std::chrono::duration<double>(dataGroupEnd - dataGroupBegin).count());
  }

  error = H5Gclose(groupId);
  RETURN_ON_ERROR;
  error = H5Fclose(fileId);
  RETURN_ON_ERROR;

  auto globalEnd = std::chrono::high_resolution_clock::now();
  measurementSecs[0] = std::chrono::duration<double>(globalEnd - globalStart).count();
  measurementSecs[1] = std::accumulate(tileSecs.cbegin(), tileSecs.cend(), 0.0) / config.numTiles;
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
  for (int tileId = 0; tileId < config.numTiles; ++tileId) {
    auto tileReadBegin = std::chrono::high_resolution_clock::now();

    std::string const dataGroupPath = std::string("/data/tile") + std::to_string(tileId);
    hid_t dataId = H5Dopen(groupId, dataGroupPath.c_str(), H5P_DEFAULT);
    error = H5Dread(dataId, H5T_IEEE_F32LE, H5S_ALL, H5S_ALL, H5P_DEFAULT, readData.get());
    error = H5Dclose(dataId);
    RETURN_ON_ERROR;
    
    auto tileReadEnd = std::chrono::high_resolution_clock::now();
    readTileSecs.push_back(std::chrono::duration<double>(tileReadEnd - tileReadBegin).count());
  }

  // Read a Roi (size (z,y,x) 256x128x128) defined as a hyperslab crossing two tiles.
  std::array<uint64_t, 3> roiOrigin1{ config.tileDims[0]-128, 128, 128}, roiOrigin2{0, 128, 128};
  std::array<uint64_t, 3> roiDim{ 128, 128, 128 };
  readData.reset(new float[roiDim[0]*roiDim[1]*roiDim[2]*2]);
  uint64_t const readDataOffset = roiDim[0] * roiDim[1] * roiDim[2];
  for (int tileId = 0; tileId + 1 < config.numTiles; ++tileId) {
    auto roiReadBegin = std::chrono::high_resolution_clock::now();

    hid_t memorySpaceId = H5Screate_simple(3, roiDim.data(), NULL);

    std::string const dataGroupPath1 = std::string("/data/tile") + std::to_string(tileId);
    hid_t dataId1 = H5Dopen(groupId, dataGroupPath1.c_str(), H5P_DEFAULT);
    hid_t dataSpaceId1 = H5Dget_space(dataId1);
    error = H5Sselect_hyperslab(dataSpaceId1, H5S_SELECT_SET, roiOrigin1.data(), NULL, roiDim.data(), NULL);
    error = H5Dread(dataId1, H5T_IEEE_F32LE, memorySpaceId, dataSpaceId1, H5P_DEFAULT, readData.get());
    error = H5Dclose(dataId1);
    error = H5Sclose(dataSpaceId1);
    RETURN_ON_ERROR;

    std::string const dataGroupPath2 = std::string("/data/tile") + std::to_string(tileId+1);
    hid_t dataId2 = H5Dopen(groupId, dataGroupPath2.c_str(), H5P_DEFAULT);
    hid_t dataSpaceId2 = H5Dget_space(dataId2);
    error = H5Sselect_hyperslab(dataSpaceId2, H5S_SELECT_SET, roiOrigin1.data(), NULL, roiDim.data(), NULL);
    error = H5Dread(dataId2, H5T_IEEE_F32LE, memorySpaceId, dataSpaceId2, H5P_DEFAULT, readData.get() + readDataOffset);
    error = H5Dclose(dataId2);
    error = H5Sclose(dataSpaceId2);
    RETURN_ON_ERROR;

    H5Sclose(memorySpaceId);
    RETURN_ON_ERROR;

    auto roiReadEnd = std::chrono::high_resolution_clock::now();
    readRoiSecs.push_back(std::chrono::duration<double>(roiReadEnd - roiReadBegin).count());
  }

  // Read each XY slice.
  roiOrigin1[1] = 0; roiOrigin1[2] = 0;
  roiDim = config.tileDims;
  roiDim[0] = 1;
  readData.reset(new float[config.tileDims[1]*config.tileDims[2]]);
  for (int z = 0; z < config.tileDims[0]*config.numTiles; ++z) {
    auto xyReadBegin = std::chrono::high_resolution_clock::now();

    int tileId  = z / config.tileDims[0];
    int zInTile = z % config.tileDims[0];
    roiOrigin1[0] = zInTile;

    std::string const dataGroupPath1 = std::string("/data/tile") + std::to_string(tileId);
    hid_t dataId1 = H5Dopen(groupId, dataGroupPath1.c_str(), H5P_DEFAULT);
    hid_t dataSpaceId = H5Dget_space(dataId1);
    hid_t memorySpaceId = H5Screate_simple(3, roiDim.data(), NULL);
    error = H5Sselect_hyperslab(dataSpaceId, H5S_SELECT_SET, roiOrigin1.data(), NULL, roiDim.data(), NULL);
    error = H5Dread(dataId1, H5T_IEEE_F32LE, memorySpaceId, dataSpaceId, H5P_DEFAULT, readData.get());
    error = H5Sclose(memorySpaceId);
    error = H5Sclose(dataSpaceId);
    error = H5Dclose(dataId1);
    RETURN_ON_ERROR;

    auto xyReadEnd = std::chrono::high_resolution_clock::now();
    readXYSecs.push_back(std::chrono::duration<double>(xyReadEnd - xyReadBegin).count());
  }

  // Read each XZ slice.
  roiOrigin1.fill(0);
  roiDim = config.tileDims;
  roiDim[1] = 1;
  uint64_t roiSize = roiDim[0] * roiDim[1] * roiDim[2];
  readData.reset(new float[config.tileDims[0]*config.numTiles*config.tileDims[2]]);
  for (int y = 0; y < config.tileDims[1]; ++y) {
    auto xzReadBegin = std::chrono::high_resolution_clock::now();

    roiOrigin1[1] = y;
    hid_t memorySpaceId = H5Screate_simple(3, roiDim.data(), NULL);

    // read config.numTiles a XZ slice from reach dataset
    for (int tileId = 0; tileId < config.numTiles; ++tileId) {
      std::string const dataGroupPath1 = std::string("/data/tile") + std::to_string(tileId);
      hid_t dataId1 = H5Dopen(groupId, dataGroupPath1.c_str(), H5P_DEFAULT);
      hid_t dataSpaceId = H5Dget_space(dataId1);
      error = H5Sselect_hyperslab(dataSpaceId, H5S_SELECT_SET, roiOrigin1.data(), NULL, roiDim.data(), NULL);
      error = H5Dread(dataId1, H5T_IEEE_F32LE, memorySpaceId, dataSpaceId, H5P_DEFAULT, readData.get() + tileId * roiSize);
      error = H5Sclose(dataSpaceId);
      error = H5Dclose(dataId1);
    }

    error = H5Sclose(memorySpaceId);
    RETURN_ON_ERROR;

    auto xzReadEnd = std::chrono::high_resolution_clock::now();
    readXZSecs.push_back(std::chrono::duration<double>(xzReadEnd - xzReadBegin).count());
  }

  // Read each XZ slice.
  roiOrigin1.fill(0);
  roiDim = config.tileDims;
  roiDim[2] = 1;
  roiSize = roiDim[0] * roiDim[1] * roiDim[2];
  readData.reset(new float[config.tileDims[0]*config.numTiles*config.tileDims[1]]);
  for (int x = 0; x < config.tileDims[0]; ++x) {
    auto yzReadBegin = std::chrono::high_resolution_clock::now();

    roiOrigin1[2] = x;
    hid_t memorySpaceId = H5Screate_simple(3, roiDim.data(), NULL);

    // read config.numTiles a XZ slice from reach dataset
    for (int tileId = 0; tileId < config.numTiles; ++tileId) {
      std::string const dataGroupPath1 = std::string("/data/tile") + std::to_string(tileId);
      hid_t dataId1 = H5Dopen(groupId, dataGroupPath1.c_str(), H5P_DEFAULT);
      hid_t dataSpaceId = H5Dget_space(dataId1);
      error = H5Sselect_hyperslab(dataSpaceId, H5S_SELECT_SET, roiOrigin1.data(), NULL, roiDim.data(), NULL);
      error = H5Dread(dataId1, H5T_IEEE_F32LE, memorySpaceId, dataSpaceId, H5P_DEFAULT, readData.get() + tileId * roiSize);
      error = H5Sclose(dataSpaceId);
      error = H5Dclose(dataId1);
    }

    error = H5Sclose(memorySpaceId);
    RETURN_ON_ERROR;

    auto yzReadEnd = std::chrono::high_resolution_clock::now();
    readYZSecs.push_back(std::chrono::duration<double>(yzReadEnd - yzReadBegin).count());
  }

  error = H5Gclose(groupId);
  error = H5Fclose(fileId);
  RETURN_ON_ERROR;

  //Read (slice XY),Read (slice XZ),Read (slice YZ)
  auto mean = [](std::vector<double> const& measurements) {
    return std::accumulate(measurements.cbegin(), measurements.cend(), 0.0) / measurements.size();
  };

  measurementsSecs[0] = mean(readTileSecs);
  measurementsSecs[1] = mean(readRoiSecs);
  measurementsSecs[2] = mean(readXYSecs);
  measurementsSecs[3] = mean(readXZSecs);
  measurementsSecs[4] = mean(readYZSecs);
}

