#ifndef KNOWBASE_H
#define KNOWBASE_H

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graph_utility.hpp>
#include <boost/graph/graph_traits.hpp>
//#include <boost/config.hpp>
#include <boost/graph/graphml.hpp>
//#include <boost/tokenizer.hpp>
//#include <iostream>
//#include <iterator>
//#include <algorithm>
//#include <fstream>
//#include <limits>
//#include "global.h"
#include "hardware.h"
//#include "kernel.h"
#include "model.h"
#include "step.h"
#include "kernel_map.h"

#include <string>
#include <vector>

typedef struct vertex_properties
{
	bool is_step = false, is_kernel = false, is_hardware = false;
	int id;
	Kernel_t *kernel;
	Hardware_t *hardware;
	Step_t *step;
}vertex_properties_t;

typedef struct edge_properties
{
	bool is_performance_model = false, is_kernel_map = false;
	int id;
	Performance_model_t *performance_model;
	Kernel_map_t *kernel_map;
}edge_properties_t;

typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS, vertex_properties_t, edge_properties_t> graph_t;
typedef boost::graph_traits<graph_t>::vertex_descriptor vertex_descriptor_t;
typedef graph_t::edge_descriptor edge_descriptror_t;
typedef boost::graph_traits<graph_t>::vertex_iterator vertex_iter;
typedef boost::graph_traits<graph_t>::edge_iterator edge_iter;


void save_graph(graph_t &g, std::string filename);
std::vector<vertex_descriptor_t> load_graph(graph_t &g,
				std::string filename);
std::vector<float> select_variant(float input_dimension, graph_t KB,
		const std::string &candidates_filename);

#endif // KNOWBASE_H

