#include <boost/python.hpp>


using namespace boost::python;

struct World
{
    World(std::string msg): msg(msg) {} // added constructor
    void set(std::string msg) {
//      theia::Reconstruction reconstruction;
//      CHECK(theia::ReadReconstruction("", &reconstruction));
    }
    std::string greet() { return msg; }
    std::string msg;
    std::string const name;
    float value;
};


BOOST_PYTHON_MODULE(hello)
{
    class_<World>("World", init<std::string>())
        .def("greet", &World::greet)
        .def("set", &World::set)
        .def_readonly("name", &World::name)
      .def_readwrite("value", &World::value);
    ;
}
