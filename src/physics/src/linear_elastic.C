// Linear_Elastic.h Class for Linear Elastic Solver within GRINS

#ifndef GRINS_LINEAR_ELASTIC_C
#define GRINS_LINEAR_ELASTIC_C

// GRINS
#include "grins/solid_mechanics_abstract.h"
#include "grins/assembly_context.h"
#include "grins/material_parsing"
#include "grins_config.h"
#include "grins/math_constants.h"
#include "grins/assembly_context.h"
#include "grins/generic_ic_handler.h"
#include "grins/postprocessed_quantities.h"
#include "grins/elasticity_tensor.h"



// libmesh
#include "libmesh/getpot.h"
#include "libmesh/quadrature.h"
#include "libmesh/fem_system.h"
#include "libmesh/boundary_info.h"
#include "libmesh/elem.h"
#include "libmesh/fe_base.h"

namespace GRINS
{

	class LinearElastic : public SolidMechanicsAbstract

			      LinearElastic::LinearElastic( const GRINS::PhysicsName& physics_name, const GetPot& input)

				      : LinearElastic(physics_name,is_compressible)
				      {
					      this->_ic_handler = new GenericICHandler(physics_name,input)	
				      }
		
	void LinearElastic::init_context(AssemblyContext& context)
		{
			this->get_fe(context)->get_JxW();
			this->get_fe(context)->get_phi();
			this->get_fe(context)->get_dphi();

		}

		void LinearElastic::mass_residual_impl(bool compute_jacobian,AssembleContext& context )
		{
			const unsigned int n_u_dofs = context.get_dof_indicies(_disp_vars.u()).size();

			const std::vector<libMesh::Real> &JxW = this->get_fe(context)->get_phi();

			// Residuals that is being populated
			libMesh::DenseSubVector<libMesh::Number> &Fu = context.get_elem_resisual(_disp_vars.u());
			libMesh::DenseSubVector<libMesh::Number> &Fv = context.get_elem_resisual(_disp_vars.v());
			libMesh::DenseSubVector<libMesh::Number> &Fw = context.get_elem_resisual(_disp_vars.w());

			libMesh::DenseSubMatrix<libMesh::Number>& Kuu = context.get_elem_jacobian(_disp_vars.u(),_disp_vars.u());
			libMesh::DenseSubMatrix<libMesh::Number>& Kvv = context.get_elem_jacobian(_disp_vars.v(),_disp_vars.v());
			libMesh::DenseSubMatrix<libMesh::Number>& Kww = context.get_elem_jacobian(_disp_vars.w(),_disp_vars.w());

			unsigned int n_qpoints = context.get_element_qrule().n_points();

			for (unsigned int qp=0; qp != n_qpoints; qp++)
			{
				libMesh::Real jac = JxW[qp];

				libMesh::Real u_ddot = 0.0; 
				libMesh::Real v_ddot = 0.0;
				libMesh::Real w_ddot = 0.0;
				(interior_accel)( _disp_vars.u(),qp, u_ddot );
				(interior_accel)( _disp_vars.v(),qp, v_ddot );
				(interior_accel)( _disp_vars.w(),qp, w_ddot );

				for (unsigned int i=0; i != n_u_dofs; i++)
				{
					Fu(i) += this->_rho*_h0*u_ddot*u_phi[i][qp]*jac;
					Fv(i) += this->_rho*_h0*v_ddot*u_phi[i][qp]*jac;
					Fw(i) += this->_rho*_h0*w_ddot*u_phi[i][qp]*jac;

					if( compute_jacobian )
					{
						for (unsigned int j=0; j != n_u_dofs; j++)
						{
							libMesh::Real jac_term = this->_rho*_h0*u_phi[i][qp]*u_phi[j][qp]*jac;
							jac_term *= get_elem_solution_accel_derivative();

							Kuu(i,j) += jac_term;
							Kvv(i,j) += jac_term;
							Kww(i,j) += jac_term;
						}
					}
				}
			}

		}// end mass_residual_impl


		void LinearElastic::element_time_derivative
		(bool compute_jacobian, AssemblyContext & context)
		{
			const unsigned int n_u_dofs = context.get_dof_indicies(this->_disp_vars.u()).size();

			const std::vector<libmesh::Real> &JxW = this->get_fe(context)->get_JxW();

			//Residuals being populated
			libMesh::DenseSubVector<libMesh::Number> &Fu = context.get_elem_residual(this->_disp_vars.u());
			libMesh::DenseSubVector<libMesh::Number> &Fv = context.get_elem_residual(this->_disp_vars.v());
			libMesh::DenseSubVector<libMesh::Number> &Fw = context.get_elem_residual(this->_disp_vars.w());

			libMesh::DenseSubMatrix<libMesh::Number>& Kuu = context.get_elem_jacobian(this->_disp_vars.u(),this->_disp_vars.u());
			libMesh::DenseSubMatrix<libMesh::Number>& Kuv = context.get_elem_jacobian(this->_disp_vars.u(),this->_disp_vars.v());
			libMesh::DenseSubMatrix<libMesh::Number>* Kuw = context.get_elem_jacobian(this->_disp_vars.u(),this->_disp_vars.w());

			libMesh::DenseSubMatrix<libMesh::Number>& Kvu = context.get_elem_jacobian(this->_disp_vars.v(),this->_disp_vars.u());
			libMesh::DenseSubMatrix<libMesh::Number>& Kvv = context.get_elem_jacobian(this->_disp_vars.v(),this->_disp_vars.v());
			libMesh::DenseSubMatrix<libMesh::Number>* Kvw = context.get_elem_jacobian(this->_disp_vars.v(),this->_disp_vars.w());

			libMesh::DenseSubMatrix<libMesh::Number>* Kwu = context.get_elem_jacobian(this->_disp_vars.w(),this->_disp_vars.u());
			libMesh::DenseSubMatrix<libMesh::Number>* Kwv = context.get_elem_jacobian(this->_disp_vars.w(),this->_disp_vars.v());
			libMesh::DenseSubMatrix<libMesh::Number>* Kww = context.get_elem_jacobian(this->_disp_vars.w(),this->_disp_vars.w());

			unsigned int n_qpoints = context.get_element_qrule().n_points();

			// Getting phi and dphi
			const std::vector<std::vector<libMesh::Real>> & phi = this->get_fe(context)->get_phi();
			const std::vector<libmesh::RealGradient> & grad_phi = this->get_fe(context)->get_dphi();

			for (unsigned in qp = 0; qp != n_qpoints; qp++)
			{
				// Gradients are w.r.t. master element coordinates
				libmesh::Gradient grad_u grad_v grad_w;
				grad_u = interior_gradient(disp_vars.u(),qp);
				grad_v = interior_gradient(disp_vars.v(),qp);
				grad_w = interior_gradient(disp_vars.v(),qp);

				Tensor grad_U (grad_u,grad_v,grad_w);						
				Tensor tau

					for (unsigned int m = 0; m < n_u_dofs; m++ )
					{
						for( unsigned int n = 0; n < n_u_dofs; n++ )
						{
							for( unsigned int o = 0; o < n_u_dofs; o++)
							{
								for( unsigned int p = 0; p < n_u_dofs; p++)
								{
									tau(m,n) = C(m,n,o,p)*grad_U(o,p);	
								}
							}
						}
					}


				for (unsigned int  i=0; i != n_u_dofs; i++)
				{

					for(unsigned int alpha = 0; alpha < part_dim; alpha++)
					{

						Fu(i) += tau(0,alpha)*grad_phi[i][qp](alpha)*JxW[qp];

						Fv(i) += tau(1,alpha)*grad_phi[i][qp](alpha)*JxW[qp];

						Fw(i) += tau(2,alpha)*grad_phi[i][qp](alpha)*JxW[qp];

						if( compute_jacobian )
						{
							for (unsigned int j=0; j != n_u_dofs; j++)
							{
								for (unsigned int beta = 0; beta < 3; beta++)
								{
									// Convenience
									const Real c0 = grad_phi[j][qp](beta)*c.get_elem_solution_derivative();

									libMesh::Real dtau_uu = elasticity_tensor(0, alpha, 0, beta)*c0;
									libMesh::Real dtau_uv = elasticity_tensor(0, alpha, 1, beta)*c0;
									libMesh::Real dtau_uw = elasticity_tensor(0, alpha, 2, beta)*c0;
									libMesh::Real dtau_vu = elasticity_tensor(1, alpha, 0, beta)*c0;
									libMesh::Real dtau_vv = elasticity_tensor(1, alpha, 1, beta)*c0;
									libMesh::Real dtau_vw = elasticity_tensor(1, alpha, 2, beta)*c0;
									libMesh::Real dtau_wu = elasticity_tensor(2, alpha, 0, beta)*c0;
									libMesh::Real dtau_wv = elasticity_tensor(2, alpha, 1, beta)*c0;
									libMesh::Real dtau_ww = elasticity_tensor(2, alpha, 2, beta)*c0;

									Kuu(i,j) += dtau_uu*grad_phi[i][qp](alpha)*JxW[qp];
									Kuv(i,j) += dtau_uv*grad_phi[i][qp](alpha)*JxW[qp];
									Kuw(i,j) += dtau_uw*grad_phi[i][qp](alpha)*JxW[qp];
									Kvu(i,j) += dtau_vu*grad_phi[i][qp](alpha)*JxW[qp];
									Kvv(i,j) += dtau_vv*grad_phi[i][qp](alpha)*JxW[qp];
									Kvw(i,j) += dtau_vw*grad_phi[i][qp](alpha)*JxW[qp];
									Kwu(i,j) += dtau_wu*grad_phi[i][qp](alpha)*JxW[qp];
									Kwv(i,j) += dtau_wv*grad_phi[i][qp](alpha)*JxW[qp];
									Kww(i,j) += dtau_ww*grad_phi[i][qp](alpha)*JxW[qp];
								}
							}
						} // end if compute jacobian
					} // end for alpha
				} // end for i
			} // end for loop through qp
		} //end element time derivative			
libMesh::Real LinearElastic::C(unsigned int i, unsigned int j, unsigned int k; unsigned int l)
	{
		MaterialParsing::read_property( input,"E",(*this), E);
		MaterialParsing::read_property( input,"nu",(*this),nu);
		
		// Define the Lame constants
		  const Real lambda_1 = (E*nu)/((1.+nu)*(1.-2.*nu));
		  const Real lambda_2 = E/(2.*(1.+nu));
		
		       return
		           lambda_1 * kronecker_delta(i, j) * kronecker_delta(k, l) +
	           lambda_2 * (kronecker_delta(i, k) * kronecker_delta(j, l) + kronecker_delta(i, l) * kronecker_delta(j, k));
		
	} //end C

libMesh::Real LinearElastic::kronecker_delta(unsigned int i, unsigned int j)
	{
		if(i == j)
		{
			return 1.0;
		}
		else
		{
			return 0.0;
		}
	} // end kronecker_delta

} // end class LinearElastic

} // end namespace GRINS
#endif // GRINS_LINEAR_ELASTIC_C
