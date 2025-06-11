
#include <array>
#include <cstdlib>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <vector>

#include "configuration.h"
#include "C_multiple_datasets.h"
#include "C_single_dataset.h"

typedef void(*read_write_fun_t)(Configuration const& config, double* measuredSecs);

static constexpr double MiB = 1ull << 20;

double throughputMiBps(size_t dataSize, double secs) {
  return static_cast< double >(dataSize) / MiB / secs;
}

void benchmark(std::string const& ID, read_write_fun_t read_fun, read_write_fun_t write_fun, std::ostream& measurements, Configuration const& config) {
  std::array<double, 7> times;
  write_fun(config, times.data());
  double const fileSizeMiB = static_cast<double>(std::filesystem::file_size(config.getFilename())) / MiB;
  read_fun(config, times.data() + 2);

  auto const gs = config.globalShape();
  auto const sliceXYSize = gs[1] * gs[2] * sizeof(float);
  auto const sliceXZSize = gs[0] * gs[2] * sizeof(float);
  auto const sliceYZSize = gs[0] * gs[1] * sizeof(float);
  auto const tileSize = config.tileDims[0] * config.tileDims[1] * config.tileDims[2] * sizeof(float);
  auto const roiSize = 128 * 128 * 128 * sizeof(float);

  measurements <<
  formatedString(
    "%3s,%3d,%3d,%3d,%d,%.1E,"/*write*/"%.1E,%.1E,%.1E,"/*read*/"%.1E,%.1E,%.1E,%.1E,%.1E,%.1E,%.1E,%.1E,%.1E,%.1E",
    ID, config.tileDims[0], config.numTiles[0], (config.chunkSizes ? config.chunkSizes->at(0) : 0), config.random, fileSizeMiB,
    times[0], times[1], throughputMiBps(tileSize, times[1]),
    times[2], throughputMiBps(tileSize, times[2]),
    times[3], throughputMiBps(roiSize, times[3]),
    times[4], throughputMiBps(sliceXYSize, times[4]),
    times[5], throughputMiBps(sliceXZSize, times[5]),
    times[6], throughputMiBps(sliceYZSize, times[6])
  );
  measurements << std::endl;
}

void run_benchmark(std::string const& ID, read_write_fun_t read_fun, read_write_fun_t write_fun, std::ostream& measurements, std::vector<Configuration> const& configurations) {
  for (auto const& config : configurations) {
    benchmark(ID, read_fun, write_fun, measurements, config);
  }
}

int main(int argc, char** argv) {

  std::string const pathPrefix = "C:/tmpdata/";

  std::vector<std::array<uint64_t, 3>> tileDimsVec{
    {256,256,256}, //{512,512,512}
  };
  std::vector<std::array<uint64_t, 3>> numTilesVec{
    {2,2,2}, //{8,8,8},
  };
  std::vector<std::array<uint64_t, 3>> chunkSizesVec{
    {16,16,16}, {32,32,32},// {64,64,64}, {128,128,128}//, {256,256,256}
  };
  std::vector<std::array<uint64_t, 3>*> chunkSizesPtrVec;
  //chunkSizesPtrVec.push_back(nullptr);
  for (auto& cs : chunkSizesVec) chunkSizesPtrVec.push_back(std::addressof(cs));

  std::vector<Configuration> configurations;
  for (auto const& tileDims : tileDimsVec) {
    for (auto const& numTiles : numTilesVec) {
      for (auto& chunkSizesPtr : chunkSizesPtrVec) {
        for (bool random : {false, /*true */ }) {
          Configuration config;
          config.numTiles         = numTiles;
          config.tileDims         = tileDims;
          config.random           = random;
          config.chunkSizes       = chunkSizesPtr;
          config.compressionLevel = 9;
          config.pathPrefix       = pathPrefix;
          configurations.push_back(config);
        }
      }
    }
  }


  std::ofstream measurements{"D:/h5_bench_measurements.txt"};
  measurements 
    << "Method,Tile dim,Number of tiles,Chunk size,Random?,File size [MiB],"
    << "Write ([s] total),Write ([s] avg),Write ([MiB/s] avg),"
    << "Read ([s] tile),Read ([MiB/s] tile),Read ([s] roi),Read ([MiB/s] roi),Read ([s] XY),Read ([MiB/s] XY),Read ([s] XZ),Read ([MiB/s] XZ),Read ([s] YZ),Read ([MiB/s] YZ)\n";

  run_benchmark("CMD", C_multiple_datasets_read, C_multiple_datasets_write, measurements, configurations);
  run_benchmark("CSD", C_multiple_datasets_read, C_multiple_datasets_write, measurements, configurations);

  return EXIT_SUCCESS;
}
