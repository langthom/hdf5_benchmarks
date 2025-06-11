
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

  hid_t dcplId  = H5P_DEFAULT; // dcpl == dataset creation property list
  if (config.chunkSizes) {
    dcplId = H5Pcreate(H5P_DATASET_CREATE);
    error  = H5Pset_chunk(dcplId, 3, config.chunkSizes->data());
    error  = H5Pset_fill_time(dcplId, H5D_FILL_TIME_NEVER);
    error  = H5Pset_deflate(dcplId, config.compressionLevel);
  }

  // Set caching for all datasets
  hid_t faplId = H5Pcreate(H5P_FILE_ACCESS); // file access property list
  {
    int _dummy;
    size_t n_slots, n_bytes;
    double chunkPreemption;
    H5Pget_cache(faplId, &_dummy, &n_slots, &n_bytes, &chunkPreemption);
    chunkPreemption = 1.0;
    n_bytes = config.cacheSize();
    H5Pset_cache(faplId, _dummy, n_slots, n_bytes, chunkPreemption);
  }


  // Set creation properties, leaving all meta data on all page
  hid_t fcplId = H5Pcreate(H5P_FILE_CREATE);
  {
    error = H5Pset_file_space_strategy(fcplId, H5F_FSPACE_STRATEGY_PAGE, true, 256);
    error = H5Pset_file_space_page_size(fcplId, 4096);
  }

  hid_t fileId  = H5Fcreate(targetH5File.c_str(), H5F_ACC_TRUNC, fcplId, faplId);
  hid_t groupId = H5Gcreate2(fileId, "data", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

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
  error = H5Pclose(fcplId);
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

  // Set caching for all datasets
  hid_t faplId = H5Pcreate(H5P_FILE_ACCESS); // file access property list
  {
    int _dummy;
    size_t n_slots, n_bytes;
    double chunkPreemption;
    H5Pget_cache(faplId, &_dummy, &n_slots, &n_bytes, &chunkPreemption);
    chunkPreemption = 1.0;
    n_bytes = config.cacheSize();
    H5Pset_cache(faplId, _dummy, n_slots, n_bytes, chunkPreemption);
  }

  hid_t fileId  = H5Fopen(fname.c_str(), H5F_ACC_RDONLY, faplId);
  hid_t groupId = H5Gopen(fileId, "/data", H5P_DEFAULT);
  
  // Read full tiles
  readData.reset(new float[config.tileDims[0] * config.tileDims[1] * config.tileDims[2]]);

  for (int tileZ = 0; tileZ < config.numTiles[0]; ++tileZ) {
    for (int tileY = 0; tileY < config.numTiles[1]; ++tileY) {
      for (int tileX = 0; tileX < config.numTiles[2]; ++tileX) {
        auto tileReadBegin = std::chrono::high_resolution_clock::now();

        std::string const dataGroupPath = std::string("/data/tile") + std::to_string(tileZ) + std::to_string(tileY) + std::to_string(tileX);
        hid_t dataId = H5Dopen2(groupId, dataGroupPath.c_str(), H5P_DEFAULT);
        error = H5Dread(dataId, H5T_IEEE_F32LE, H5S_ALL, H5S_ALL, H5P_DEFAULT, readData.get());
        error = H5Dclose(dataId);
        RETURN_ON_ERROR;

        auto tileReadEnd = std::chrono::high_resolution_clock::now();
        readTileSecs.push_back(std::chrono::duration<double>(tileReadEnd - tileReadBegin).count());
      }
    }
  }


  // Read a Roi of shape 128x128x128 crossing into 8 tiles.
  // Corresponds to a 64x64x64 Roi in each tile.
  {
    auto globalRoiData = std::make_unique<float[]>(128 * 128 * 128);
    std::array<uint64_t, 3> localRoiDim{64, 64, 64};
    auto localRoiReadData = std::make_unique<float[]>(localRoiDim[0]*localRoiDim[1]*localRoiDim[2]);
    hid_t localRoiMemSpaceId = H5Screate_simple(3, localRoiDim.data(), NULL);

    auto __read = [&](int tileZ, int tileY, int tileX, std::array<uint64_t, 3> const& localRoiOrigin) {
      std::string const datasetPath = std::string("/data/tile") + std::to_string(tileZ) + std::to_string(tileY) + std::to_string(tileX);

      hid_t datasetId = H5Dopen(groupId, datasetPath.c_str(), H5P_DEFAULT);
      hid_t filespaceId = H5Dget_space(datasetId);

      error = H5Sselect_hyperslab(filespaceId, H5S_SELECT_SET, localRoiOrigin.data(), NULL, localRoiDim.data(), NULL);
      error = H5Dread(datasetId, H5T_NATIVE_FLOAT, localRoiMemSpaceId, filespaceId, H5P_DEFAULT, localRoiReadData.get());
      error = H5Dclose(datasetId);
      error = H5Sclose(filespaceId);

      // copy to target
      auto tileStart = globalRoiData.get() + ((tileZ * 128) + tileY) * 128 + tileX;
      for (int z = 0; z < 64; ++z) {
        for (int y = 0; y < 64; ++y) {
          auto d = tileStart + (z * 128 + y) * 128;
          auto s = localRoiReadData.get() + ((z*64)+y)*64;
          std::memcpy(d, s, 64 * sizeof(float));
        }
      }
    };

    auto roiReadStart = std::chrono::high_resolution_clock::now();
    __read(0, 0, 0, {config.tileDims[0]-64, config.tileDims[1]-64, config.tileDims[2]-64});
    __read(0, 0, 1, {config.tileDims[0]-64, config.tileDims[1]-64,                     0});
    __read(0, 1, 0, {config.tileDims[0]-64,                     0, config.tileDims[2]-64});
    __read(0, 1, 1, {config.tileDims[0]-64,                     0,                     0});
    __read(1, 0, 0, {0,                     config.tileDims[1]-64, config.tileDims[2]-64});
    __read(1, 0, 1, {0,                     config.tileDims[1]-64,                     0});
    __read(1, 1, 0, {0,                     0,                     config.tileDims[2]-64});
    __read(1, 1, 1, {0,                     0,                                         0});
    auto roiReadEnd = std::chrono::high_resolution_clock::now();
    readRoiSecs.push_back(std::chrono::duration<double>(roiReadEnd - roiReadStart).count());

    H5Sclose(localRoiMemSpaceId);
  }

  auto readSlice = [&](int orientation, std::vector<double>& timesSecs) {
    std::array<uint64_t, 3> localOrigin{0, 0, 0};
    std::array<uint64_t, 3> localDim  = config.tileDims;
    std::array<uint64_t, 3> globalDim = config.globalShape();
    std::array<uint64_t, 3> index;

    uint64_t const numSlices = globalDim[orientation];
    uint64_t numTiles1, numTiles2, localRowDim, localColDim;

    switch (orientation) {
    case 0/*XY*/: numTiles1 = config.numTiles[1], numTiles2 = config.numTiles[2], localRowDim = 1, localColDim = 2; break;
    case 1/*XZ*/: numTiles1 = config.numTiles[0], numTiles2 = config.numTiles[2], localRowDim = 0, localColDim = 2; break;
    case 2/*YZ*/: numTiles1 = config.numTiles[0], numTiles2 = config.numTiles[1], localRowDim = 0, localColDim = 1; break;
    }

    localDim[orientation] = 1;
    globalDim[orientation] = 1;
    auto localReadData   = std::make_unique<float[]>( localDim[0] *  localDim[1] *  localDim[2]);
    auto globalSliceData = std::make_unique<float[]>(globalDim[0] * globalDim[1] * globalDim[2]);


    hid_t localSliceMemorySpaceId = H5Screate_simple(3, localDim.data(), NULL);

    for (uint64_t sliceIx = 0; sliceIx < numSlices; sliceIx += 32) {
      auto sliceStart = std::chrono::high_resolution_clock::now();
      auto sliceTile   = sliceIx / config.tileDims[orientation];
      auto sliceInTile = sliceIx % config.tileDims[orientation];
      localOrigin[orientation] = sliceInTile;

      for (uint64_t rowTileIx = 0; rowTileIx < numTiles1; ++rowTileIx) {
        for (uint64_t colTileIx = 0; colTileIx < numTiles2; ++colTileIx) {
          switch (orientation) {
          case 0/*XY*/: index[0] = sliceTile, index[1] = rowTileIx, index[2] = colTileIx; break;
          case 1/*XZ*/: index[0] = rowTileIx, index[1] = sliceTile, index[2] = colTileIx; break;
          case 2/*YZ*/: index[0] = rowTileIx, index[1] = colTileIx, index[2] = sliceTile; break;
          }
          std::string const datasetPath = std::string("/data/tile") + std::to_string(index[0]) + std::to_string(index[1]) + std::to_string(index[2]);

          hid_t datasetId = H5Dopen(groupId, datasetPath.c_str(), H5P_DEFAULT);
          hid_t filespaceId = H5Dget_space(datasetId);

          error = H5Sselect_hyperslab(filespaceId, H5S_SELECT_SET, localOrigin.data(), NULL, localDim.data(), NULL);
          error = H5Dread(datasetId, H5T_NATIVE_FLOAT, localSliceMemorySpaceId, filespaceId, H5P_DEFAULT, localReadData.get());
          error = H5Dclose(datasetId);
          error = H5Sclose(filespaceId);

          // need to manually copy the locally read data into the correct position in the global slice.
          // we implement this to make a fair comparison against other methods
          for (int _row = 0; _row < localDim[localRowDim]; ++_row) {
            auto d = globalSliceData.get() + (rowTileIx + _row) * globalDim[localRowDim] + colTileIx * globalDim[localColDim];
            auto s = localReadData.get() + _row * localDim[localColDim];
            std::memcpy(d, s, localDim[localColDim] * sizeof(float));
          }
        }
      }

      auto sliceEnd = std::chrono::high_resolution_clock::now();
      timesSecs.push_back(std::chrono::duration<double>(sliceEnd - sliceStart).count());
    }

    error = H5Sclose(localSliceMemorySpaceId);
  };

  readSlice(0/*XY*/, readXYSecs);
  readSlice(1/*XZ*/, readXZSecs);
  readSlice(2/*YZ*/, readYZSecs);

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

