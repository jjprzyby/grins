// Linear_Elastic.h Class for Linear ver

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
				      this->get_fe(context)->get_dphidxi();
				      this->get_fe(context)->get_dphideta();

				      // Need for constructing metric tensors
				      this->get_fe(context)->get_dxyzdxi();
				      this->get_fe(context)->get_dxyzdeta();
				      this->get_fe(context)->get_dxidx();
				      this->get_fe(context)->get_dxidy();
				      this->get_fe(context)->get_dxidz();
				      this->get_fe(context)->get_detadx();
				      this->get_fe(context)->get_detady();
				      this->get_fe(context)->get_detadz();
			      }

	LinearElastic::init_variables( libMesh::FEMSystem* system )
		{

		}

	LinearElastic::mass_residual_impl(bool compute_jacobian,AssembleContext& context, InterriorFuncType interior_solution, VarDerivType get_sol_deriv, libMesh::Real mu )
		{

		}

	LinearElastic::compute_metric_tensors(unsigned int qp, const libMesh::FEBase& elem, const AssemblyContext& context, const libMesh::Gradient& grad_u, const libMesh::Gradient& grad_v, const libMesh& grad_w, 
					      libMesh::TensorValue<libMesh::Real>& a_cov, libMesh::TensorValue<libMesh::Real>& a_contra, libMesh::TensorValue<libMesh::Real>& A_cov, libMesh::TensorValue<libMesh::Real>& A_contra,
					      libmesh::Real& lambda_sq )
		{

		}	

	LinearElastic::register_postprocessingvars(const GetPot& input, PostProcessedQuatitities<libMesh::Reak>& postprocessing)
		{

		}


	LinearElastic::element_time_derivative
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

			// All shape function gradients are w.r.t. master element coordinates
			const std::vector<std::vector<libMesh::Real> >& dphi_dxi =
				this->get_fe(context)->get_dphidxi();

			const std::vector<std::vector<libMesh::Real> >& dphi_deta =
				this->get_fe(context)->get_dphideta();

			const libMesh::DenseSubVector<libMesh::Number>& u_coeffs = context.get_elem_solution( this->_disp_vars.u() );
			const libMesh::DenseSubVector<libMesh::Number>& v_coeffs = context.get_elem_solution( this->_disp_vars.v() );
			const libMesh::DenseSubVector<libMesh::Number>* w_coeffs = context.get_elem_solution( this->_disp_vars.w() );

			// Need these to build up the covariant and contravariant metric tensors
			const std::vector<libMesh::RealGradient>& dxdxi  = this->get_fe(context)->get_dxyzdxi();
			const std::vector<libMesh::RealGradient>& dxdeta = this->get_fe(context)->get_dxyzdeta();


			for (unsigned in qp = 0; qp != n_qpoints; qp++)
			{
				// Gradients are w.r.t. master element coordinates
				libmesh::Gradient grad_u grad_v grad_w;

				for( unsigned int d = 0; d < n_u_dofs; d++)
				{
					libmesh::RealGradient u_gradphi( dphi_dxi[d][qp], dphi_deta[d][qp] );
					grad_u += u_coeffs(d)*u_gradphi;
					grad_v += v_coeffs(d)*v_gradphi;
					gead_w += w_coeffs(d)*w_gradphi;
				}

				libMesh::RealGradient grad_x( dxdxi[qp](0), dxdeta[qp](0) );
				libMesh::RealGradient grad_y( dxdxi[qp](1), dxdeta[qp](1) );
				libMesh::RealGradient grad_z( dxdxi[qp](2), dxdeta[qp](2) );

				libMesh::TensorValue<libMesh::Real> a_cov, a_contra, A_cov, A_contra;
				libMesh::Real lambda_sq=0;	

				this->compute_metric_tensors(qp, *(this->get_fe(context)), context, grad_v, grad_v, grad_w, a_cov, a_contra, A_cov, A_contra, lambda_sq);

				const unsigned int part_dim = 3; //The part's dimension is always 3 for this physics

				// Compute stress and elasticity tensors
				libmesh::TensorValue<libMesh::Real> tau;
				ElasticityTensor C;
				this->_stress_strain_law.compute_stress_and_elasticity(part_dim,a_contra,a_cov,A_contra,A_cov,tau,C);

				libMesh::Real jac = JxW[qp];

				for (unsigned int  i=0; i != n_u_dofs; i++)
				{
					libMesh::Real jac = JxW[qp];

					for(unsigned int alpha = 0; alpha < part_dim; alpha++)
					{
						for(unsigned int beta = 0; beta < part_dim ; beta ++)
						{
							Fu(i) += factor*( (grad_x(beta) + grad_u(beta))*u_gradphi(alpha) +
									(grad_x(alpha) + grad_u(alpha))*u_gradphi(beta) );

							Fv(i) += factor*( (grad_y(beta) + grad_v(beta))*u_gradphi(alpha) +
									(grad_y(alpha) + grad_v(alpha))*u_gradphi(beta) );


							Fw(i) += factor*( (grad_z(beta) + grad_w(beta))*u_gradphi(alpha) +
									(grad_z(alpha) + grad_w(alpha))*u_gradphi(beta) );


						}
					}

				}
				if( compute_jacobian )
				{
					for (unsigned int i=0; i != n_u_dofs; i++)
					{
						libMesh::RealGradient u_gradphi_i( dphi_dxi[i][qp], dphi_deta[i][qp] );

						for (unsigned int j=0; j != n_u_dofs; j++)
						{
							libMesh::RealGradient u_gradphi_j( dphi_dxi[j][qp], dphi_deta[j][qp] );

							for( unsigned int alpha = 0; alpha < manifold_dim; alpha++ )
							{
								for( unsigned int beta = 0; beta < manifold_dim; beta++ )
								{
									const libMesh::Real diag_term = 0.5*this->_h0*jac*tau(alpha,beta)*context.get_elem_solution_derivative()*
										( u_gradphi_j(beta)*u_gradphi_i(alpha) +
										  u_gradphi_j(alpha)*u_gradphi_i(beta) );
									Kuu(i,j) += diag_term;

									Kvv(i,j) += diag_term;


									Kww(i,j) += diag_term;

									for( unsigned int lambda = 0; lambda < manifold_dim; lambda++ )
									{
										for( unsigned int mu = 0; mu < manifold_dim; mu++ )
										{
											const libMesh::Real dgamma_du = 0.5*( u_gradphi_j(lambda)*(grad_x(mu)+grad_u(mu)) +
													(grad_x(lambda)+grad_u(lambda))*u_gradphi_j(mu) );

											const libMesh::Real dgamma_dv = 0.5*( u_gradphi_j(lambda)*(grad_y(mu)+grad_v(mu)) +
													(grad_y(lambda)+grad_v(lambda))*u_gradphi_j(mu) );

											const libMesh::Real C1 = 0.5*this->_h0*jac*C(alpha,beta,lambda,mu) * 
												context.get_elem_solution_derivative();

											const libMesh::Real x_term = C1*( (grad_x(beta)+grad_u(beta))*u_gradphi_i(alpha) +
													(grad_x(alpha)+grad_u(alpha))*u_gradphi_i(beta) );

											const libMesh::Real y_term = C1*( (grad_y(beta)+grad_v(beta))*u_gradphi_i(alpha) +
													(grad_y(alpha)+grad_v(alpha))*u_gradphi_i(beta) );

											Kuu(i,j) += x_term*dgamma_du;

											Kuv(i,j) += x_term*dgamma_dv;

											Kvu(i,j) += y_term*dgamma_du;

											Kvv(i,j) += y_term*dgamma_dv;



											const libMesh::Real dgamma_dw = 0.5*( u_gradphi_j(lambda)*(grad_z(mu)+grad_w(mu)) +
													(grad_z(lambda)+grad_w(lambda))*u_gradphi_j(mu) );

											const libMesh::Real z_term = C1*( (grad_z(beta)+grad_w(beta))*u_gradphi_i(alpha) +
													(grad_z(alpha)+grad_w(alpha))*u_gradphi_i(beta) );

											Kuw(i,j) += x_term*dgamma_dw;

											Kvw(i,j) += y_term*dgamma_dw;

											Kwu(i,j) += z_term*dgamma_du;

											Kwv(i,j) += z_term*dgamma_dv;

										} // end mu for loop
									} //end lambda for loop
								} // end beta for loop
							} // end alpha for loop
						} // end j for loop
					} // end i for loop
				} // end if compute jacobian
			} // end for loop through qp
		} //end element time derivative			

	void LinearElastic::element_constraint
	(bool compute_jacobian, AssemblyContext & context)
		{
		
		}

	void LinearElastic::element_contraint(bool compute_jacobian, AssemblyContext & context)
		{

		}
	

	void LinearElastic::compute_postprocessed_quantity(unsigned int quantity_index, const AssemblyContext& context, const libMesh::Point& point, libMesh::Real& value)
		{

		}
			
} // end class LinearElastic

} // end namespace GRINS
#endif // GRINS_LINEAR_ELASTIC_C
