#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <octomap/octomap.h>
#include <sstream>

namespace py = pybind11;


octomap::OcTree* readTree(py::bytes tree_data, double res) {
    std::string data_str = static_cast<std::string>(tree_data);
    std::stringstream ss(data_str);
    
    octomap::OcTree* tree = new octomap::OcTree(res);
    if (!data_str.empty()) {
        tree->readBinaryData(ss);
    }
    return tree;
}

py::array_t<uint32_t> extract_local_grid(py::bytes tree_data, 
                                         double tx, double ty, double tz, 
                                         double qx, double qy, double qz, double qw, 
                                         double res, int grid_size, int z_layers) {
                                             
    octomap::OcTree* tree = readTree(tree_data, res);
    
    auto result = py::array_t<uint32_t>({z_layers, grid_size, grid_size});
    auto buf = result.request();
    uint32_t* ptr = static_cast<uint32_t*>(buf.ptr);

    octomath::Pose6D pose(
        octomath::Vector3(tx, ty, tz), 
        octomath::Quaternion(qw, qx, qy, qz) 
    );

    int idx = 0;
    double center_xy = grid_size / 2.0;
    double center_z = z_layers / 2.0;

    for (int z = 0; z < z_layers; ++z) {
        for (int y = 0; y < grid_size; ++y) {
            for (int x = 0; x < grid_size; ++x) {
                double local_x = (x - center_xy) * res + (res / 2.0);
                double local_y = (y - center_xy) * res + (res / 2.0);
                double local_z = (z - center_z) * res + (res / 2.0);
                
                octomath::Vector3 global_pt = pose.transform(octomath::Vector3(local_x, local_y, local_z));
                octomap::OcTreeNode* node = tree->search(global_pt);
                
                ptr[idx++] = (node != nullptr && tree->isNodeOccupied(node)) ? 1 : 0; 
            }
        }
    }
    
    delete tree;
    return result;
}


bool save_map_stateless(py::bytes tree_data, double res, const std::string& filename) {
    octomap::OcTree* tree = readTree(tree_data, res);
    bool success = tree->writeBinary(filename);
    delete tree;
    return success;
}


class OctomapManager {
public:
    OctomapManager(double res) { 
        // avoid needing to write more than just incoming points 

        tree = new octomap::OcTree(res);
        
        // stricter params than normal octomap 
        // consider all incoming points as real, 
        tree->setProbHit(0.97);
        tree->setClampingThresMax(0.97);
        tree->setProbMiss(0.12);
        tree->setClampingThresMin(0.12);
    }

    ~OctomapManager() {
        delete tree;
    }

    void inject_points(py::array_t<float> global_points) {
        auto buf = global_points.request();
        float* ptr = static_cast<float*>(buf.ptr);
        int num_points = buf.shape[0];

        for (int i = 0; i < num_points; ++i) {
            tree->updateNode(ptr[i * 3 + 0], ptr[i * 3 + 1], ptr[i * 3 + 2], true);
        }
    }

    // convert to bytes for rviz
    py::bytes get_serialized_map() {
        tree->updateInnerOccupancy(); // update before serializing
        std::stringstream ss;
        tree->writeBinaryData(ss);

        return py::bytes(ss.str());
    }

    // save as .bt file
    void save_map(const std::string& filename) {
        tree->updateInnerOccupancy();
        tree->writeBinaryConst(filename);
    }

private:
    octomap::OcTree* tree;
};



PYBIND11_MODULE(octomap_bridge, m) {
    
    m.def("extract_local_grid", &extract_local_grid);
    m.def("save_map", &save_map_stateless);

    py::class_<OctomapManager>(m, "OctomapManager")
        .def(py::init<double>())
        .def("inject_points", &OctomapManager::inject_points)
        .def("get_serialized_map", &OctomapManager::get_serialized_map)
        .def("save_map", &OctomapManager::save_map);
}