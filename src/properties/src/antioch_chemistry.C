//-----------------------------------------------------------------------bl-
//--------------------------------------------------------------------------
//
// GRINS - General Reacting Incompressible Navier-Stokes
//
// Copyright (C) 2014-2017 Paul T. Bauman, Roy H. Stogner
// Copyright (C) 2010-2013 The PECOS Development Team
//
// This library is free software; you can redistribute it and/or
// modify it under the terms of the Version 2.1 GNU Lesser General
// Public License as published by the Free Software Foundation.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
// Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc. 51 Franklin Street, Fifth Floor,
// Boston, MA  02110-1301  USA
//
//-----------------------------------------------------------------------el-


#include "grins_config.h"

#ifdef GRINS_HAVE_ANTIOCH

// This class
#include "grins/antioch_chemistry.h"

// GRINS
#include "grins/materials_parsing.h"

// Antioch
#include "antioch/default_filename.h"

// libMesh
#include "libmesh/getpot.h"

namespace GRINS
{
  AntiochChemistry::AntiochChemistry( const GetPot& input,
                                      const std::string& material )
    : ParameterUser("AntiochChemistry")
  {
    {
      std::string warning = "==============================================\n";
      warning += "WARNING: This AntiochChemistry constructor is DEPREACTED!\n";
      warning += "         Prefer alternate constructor where parsing\n";
      warning += "         is done outside this class.\n";
      warning += "==============================================\n";

      libmesh_warning(warning);
    }
    std::vector<std::string> species_list;
    MaterialsParsing::parse_chemical_species(input,material,species_list);

    bool verbose_antioch_read = input("Materials/"+material+"/GasMixture/Antioch/verbose_read",false);

    std::string species_data_filename = input("Materials/"+material+"/GasMixture/Antioch/species_data", "default" );
    if( species_data_filename == std::string("default") )
      species_data_filename = Antioch::DefaultInstallFilename::chemical_mixture();

    std::string vibration_data_filename = input("Materials/"+material+"/GasMixture/Antioch/vibration_data", "default" );
    if( vibration_data_filename == std::string("default") )
      vibration_data_filename = Antioch::DefaultInstallFilename::vibrational_data();

    std::string electronic_data_filename = input("Materials/"+material+"/GasMixture/Antioch/electronic_data", "default" );
    if( electronic_data_filename == std::string("default") )
      electronic_data_filename = Antioch::DefaultInstallFilename::electronic_data();

    // By default, Antioch is using its ASCII parser. We haven't added more options yet.
    _antioch_gas.reset( new Antioch::ChemicalMixture<libMesh::Real>( species_list,
                                                                     verbose_antioch_read,
                                                                     species_data_filename,
                                                                     vibration_data_filename,
                                                                     electronic_data_filename ) );
  }

  AntiochChemistry::AntiochChemistry
  ( libMesh::UniquePtr<Antioch::ChemicalMixture<libMesh::Real> > & chem_mixture )
    : ParameterUser("AntiochChemistry")
  {
    /*! \todo Use std::move when we have C++11 */
    _antioch_gas.reset( chem_mixture.release() );
  }

  std::string AntiochChemistry::species_name( unsigned int species_index ) const
  {
    libmesh_assert_less(species_index, _antioch_gas->n_species());

    return _antioch_gas->species_inverse_name_map().find(species_index)->second;
  }

}// end namespace GRINS

#endif // GRINS_HAVE_ANTIOCH
