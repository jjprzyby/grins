// linear_elastic.h header file for linear_elastic class

#ifndef GRINS_LINEAR_ELASTIC_H
#define GRINS_LINEAR_ELASTIC_H

// GRINS
#include "grins/solid_mechanics_abstract.h"


namespace GRINS
{
	class LinearElastic : public SolidMechanicsAbstract

	{
		public:

			LinearElastic(const PhysicsName& physics_name, const GetPot& iput);


			// Time dependent part(s) of physics for element interiors
			void init_context(AssemblyContext& context);

			void mass_residual(bool compute_jacobian,AssemblyContext& context);

			void element_time_derivative(bool compute_jacobian, AssemblyContext & context);

			libMesh::Real C(unsigned int i, unsigned int j, unsigned int k, unsigned int l);

			libMesh::Real kronecker_delta(unsigned int i, unsigned int j);

		private:

			LinearElastic();



	}; // end namespace GRINS
}
#endif // GRINS LINEAR_ELASTIC_H
