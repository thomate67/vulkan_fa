//#ifndef GRID_H
//#define GRID_H
//
//#include <glm/glm.hpp>
//#include <glm/gtc/matrix_transform.hpp>
//
//class Grid
//{
//
//
//	struct Uniform_grid_params {
//
//		Uniform_grid_params(const float grid_size_x, const float grid_size_y, const float grid_size_z,
//			const float cell_size_x, const float cell_size_y, const float cell_size_z,
//			const int cell_capacity, const glm::vec4 pos = glm::vec4(0.0f))
//			:grid_pos(pos), cell_capacity(cell_capacity)
//		{
//			const float cell_count_x = glm::ceil(grid_size_x / cell_size_x);
//			const float cell_count_y = glm::ceil(grid_size_y / cell_size_y);
//			const float cell_count_z = glm::ceil(grid_size_z / cell_size_z);
//			cell_count = glm::ivec4(cell_count_x, cell_count_y, cell_count_z, 0);
//			total_cell_count = static_cast<int>(cell_count_x * cell_count_y * cell_count_z);
//			cell_size = glm::vec4(cell_size_x, cell_size_y, cell_size_z, 0);
//			const auto half_size = glm::vec4(cell_size_x * cell_count_x, cell_size_y * cell_count_y,
//				cell_size_z * cell_count_z, 0) / 2.0f;
//			grid_min = grid_pos - half_size;
//			grid_capacity = total_cell_count * cell_capacity;
//		}
//
//
//		glm::vec4 grid_min;
//		glm::ivec4 cell_count;
//		glm::vec4 cell_size;
//		glm::vec4 grid_pos;
//		int total_cell_count;
//		int cell_capacity;
//		int grid_capacity;
//	};
//};
//
//#endif // GRID_H