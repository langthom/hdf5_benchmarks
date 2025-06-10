
#include <array>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <vector>

#include "configuration.h"
#include "C_multiple_datasets.h"
#include "C_single_dataset.h"

typedef void(*read_write_fun_t)(Configuration const& config, double* measuredSecs);

void benchmark(std::string const& ID, read_write_fun_t read_fun, read_write_fun_t write_fun, std::ostream& measurements, Configuration const& config) {
  std::array<double, 7> times;
  write_fun(config, times.data());
  read_fun(config, times.data() + 2);

  measurements << ID << "," << config.tileDims[0] << ',' << config.numTiles[0] << ',' << (config.chunkSizes ? config.chunkSizes->at(0) : 0) << ',' << config.random << ',';
  for (int i = 0; i < times.size(); ++i) {
    if (i > 0) measurements << ',';
    measurements << times[i];
  }
  measurements << '\n';
  measurements.flush();
}

int main(int argc, char** argv) {

  std::vector<std::array<uint64_t, 3>> tileDimsVec{
    {256,256,256}, //{512,512,512}
  };
  std::vector<std::array<uint64_t, 3>> numTilesVec{
    {2,2,2}, //{8,8,8},
  };
  std::vector<std::array<uint64_t, 3>> chunkSizesVec{
    {64,64,64}, //{128,128,128}, {256,256,256}
  };
  std::vector<std::array<uint64_t, 3>*> chunkSizesPtrVec;
  //chunkSizesPtrVec.push_back(nullptr);
  for (auto& cs : chunkSizesVec) chunkSizesPtrVec.push_back(std::addressof(cs));

  std::ofstream measurements{"D:/h5_bench_measurements.txt"};
  measurements << "Method,Tile dim,Number of tiles,Chunk size,Random?,Write (total),Write (avg),Read (tile),Read (roi),Read (slice XY),Read (slice XZ),Read (slice YZ)\n";

  for (auto const& tileDims : tileDimsVec) {
    for (auto const& numTiles : numTilesVec) {
      for (auto& chunkSizesPtr : chunkSizesPtrVec) {
        for (bool random : {false, true}) {

          Configuration config;
          config.numTiles    = numTiles;
          config.tileDims    = tileDims;
          config.random      = random;
          config.chunkSizes  = chunkSizesPtr;
          config.compressionLevel = 7;

          benchmark("CMD", C_multiple_datasets_read, C_multiple_datasets_write, measurements, config);
          benchmark("CSD", C_single_dataset_read, C_single_dataset_write, measurements, config);

          //std::filesystem::remove(config.getFilename());
        }
      }
    }
  }

  return EXIT_SUCCESS;
}
