
#include "C_multiple_datasets.h"
#include "configuration.h"
#include <chrono>
#include <numeric>
#include <vector>
#include "hdf5.h"

/* Consider a file of the following structure (single dataset for all tiles).

      HDF5 <filename> {
        GROUP "/" {
          GROUP "data" {
            DATASET "all_data" {
              DATATYPE H5T_NATIVE_FLOAT
              DATASPACE SIMPLE { (<global_shape> / <global_shape>) }
            }
          }
        }
      }

*/

void C_single_dataset_write(Configuration const& config, double* measurementSecs) {
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
  hid_t fileSpaceId = H5Screate_simple(3, config.globalShape().data(), NULL);
  hid_t datasetId = H5Dcreate2(groupId, "/data/all_data", H5T_NATIVE_FLOAT, fileSpaceId, H5P_DEFAULT, dcplId, H5P_DEFAULT);

  hid_t tileMemorySpaceId = H5Screate_simple(3, config.tileDims.data(), NULL);

  std::array<uint64_t, 3> origin{0, 0, 0}, dim = config.tileDims;

  for (int tileZ = 0; tileZ < config.numTiles[0]; ++tileZ) {
    for (int tileY = 0; tileY < config.numTiles[1]; ++tileY) {
      for (int tileX = 0; tileX < config.numTiles[2]; ++tileX) {
        auto dataGroupBegin = std::chrono::high_resolution_clock::now();

        origin[0] = tileZ * config.tileDims[0];
        origin[1] = tileY * config.tileDims[1];
        origin[2] = tileX * config.tileDims[2];

        error = H5Sselect_hyperslab(fileSpaceId, H5S_SELECT_SET, origin.data(), NULL, dim.data(), NULL);
        error = H5Dwrite(datasetId, H5T_NATIVE_FLOAT, tileMemorySpaceId, fileSpaceId, H5P_DEFAULT, tileData.get());

        auto dataGroupEnd = std::chrono::high_resolution_clock::now();
        tileSecs.push_back(std::chrono::duration<double>(dataGroupEnd - dataGroupBegin).count());
      }
    }
  }

  error = H5Dclose(datasetId);
  error = H5Sclose(tileMemorySpaceId);
  error = H5Sclose(fileSpaceId);
  error = H5Gclose(groupId);
  error = H5Pclose(dcplId);
  error = H5Pclose(faplId);
  error = H5Pclose(fcplId);
  error = H5Fclose(fileId);
  RETURN_ON_ERROR;

  auto globalEnd = std::chrono::high_resolution_clock::now();
  measurementSecs[0] = std::chrono::duration<double>(globalEnd - globalStart).count();
  measurementSecs[1] = std::accumulate(tileSecs.cbegin(), tileSecs.cend(), 0.0) / tileSecs.size();
}


void C_single_dataset_read(Configuration const& config, double* measurementsSecs) {
  std::string const fname = config.getFilename();
  std::vector<double> readTileSecs, readRoiSecs, readXYSecs, readXZSecs, readYZSecs;

  herr_t error;


  // Set caching for all datasets
  hid_t faplId = H5Pcreate(H5P_FILE_ACCESS); // file access property list
  {
    int _dummy;
    size_t n_slots, n_bytes;
    double chunkPreemption;
    H5Pget_cache(faplId, &_dummy, &n_slots, &n_bytes, &chunkPreemption);
    chunkPreemption = 0.75;
    n_bytes = config.cacheSize();
    H5Pset_cache(faplId, _dummy, n_slots, n_bytes, chunkPreemption);
  }

  hid_t fileId  = H5Fopen(fname.c_str(), H5F_ACC_RDONLY, faplId);
  hid_t groupId = H5Gopen2(fileId, "/data", H5P_DEFAULT);
  hid_t dataId  = H5Dopen2(groupId, "/data/all_data", H5P_DEFAULT);
  hid_t filespaceId = H5Dget_space(dataId);

  // Read full tiles
  {
    auto readTileData = std::make_unique<float[]>(config.tileDims[0] * config.tileDims[1] * config.tileDims[2]);
    hid_t tileCopyMemSpaceId = H5Screate_simple(3, config.tileDims.data(), NULL);

    for (int tileZ = 0; tileZ < config.numTiles[0]; ++tileZ) {
      for (int tileY = 0; tileY < config.numTiles[1]; ++tileY) {
        for (int tileX = 0; tileX < config.numTiles[2]; ++tileX) {
          auto tileReadBegin = std::chrono::high_resolution_clock::now();

          std::array<uint64_t, 3> origin{tileZ * config.tileDims[0], tileY * config.tileDims[1], tileX * config.tileDims[2]};
          error = H5Sselect_hyperslab(filespaceId, H5S_SELECT_SET, origin.data(), NULL, config.tileDims.data(), NULL);
          error = H5Dread(dataId, H5T_NATIVE_FLOAT, tileCopyMemSpaceId, filespaceId, H5P_DEFAULT, readTileData.get());

          auto tileReadEnd = std::chrono::high_resolution_clock::now();
          readTileSecs.push_back(std::chrono::duration<double>(tileReadEnd - tileReadBegin).count());
        }
      }
    }

    error = H5Sclose(tileCopyMemSpaceId);
  }


  // Read a Roi of shape 128x128x128 crossing into 8 tiles.
  // Corresponds to a 64x64x64 Roi in each tile.
  {
    auto roiReadStart = std::chrono::high_resolution_clock::now();

    std::array<uint64_t, 3> origin = {config.tileDims[0] - 64, config.tileDims[1] - 64, config.tileDims[2] - 64};
    std::array<uint64_t, 3> dim{128, 128, 128};
    auto roiReadData = std::make_unique<float[]>(128*128*128);

    hid_t roiReadMemSpace = H5Screate_simple(3, dim.data(), NULL);
    error = H5Sselect_hyperslab(filespaceId, H5S_SELECT_SET, origin.data(), NULL, dim.data(), NULL);
    error = H5Dread(dataId, H5T_NATIVE_FLOAT, roiReadMemSpace, filespaceId, H5P_DEFAULT, roiReadData.get());
    error = H5Sclose(roiReadMemSpace);

    auto roiReadEnd = std::chrono::high_resolution_clock::now();
    readRoiSecs.push_back(std::chrono::duration<double>(roiReadEnd-roiReadStart).count());
  }

  auto readSlice = [&](int orientation, std::vector<double>& timesSecs) {
    std::array<uint64_t, 3> sliceOrigin{0, 0, 0};
    std::array<uint64_t, 3> sliceDim = config.globalShape();
    uint64_t const numSlices = sliceDim[orientation];

    sliceDim[orientation] = 1;
    auto sliceData = std::make_unique<float[]>(sliceDim[0] * sliceDim[1] * sliceDim[2]);

    hid_t sliceCopyMemSpaceId = H5Screate_simple(3, sliceDim.data(), NULL);

    for (uint64_t sliceIx = 0; sliceIx < numSlices; sliceIx += 32) {
      sliceOrigin[orientation] = sliceIx;

      auto sliceReadStart = std::chrono::high_resolution_clock::now();

      error = H5Sselect_hyperslab(filespaceId, H5S_SELECT_SET, sliceOrigin.data(), NULL, sliceDim.data(), NULL);
      error = H5Dread(dataId, H5T_NATIVE_FLOAT, sliceCopyMemSpaceId, filespaceId, H5P_DEFAULT, sliceData.get());

      auto sliceWriteStart = std::chrono::high_resolution_clock::now();
      timesSecs.push_back(std::chrono::duration<double>(sliceWriteStart - sliceReadStart).count());
    }

    error = H5Sclose(sliceCopyMemSpaceId);
  };

  readSlice(0/*XY*/, readXYSecs);
  readSlice(1/*XZ*/, readXZSecs);
  readSlice(2/*YZ*/, readYZSecs);

  error = H5Sclose(filespaceId);
  error = H5Dclose(dataId);
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

