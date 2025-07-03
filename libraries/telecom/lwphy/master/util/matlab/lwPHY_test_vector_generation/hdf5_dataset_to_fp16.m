%%
 % Copyright (c) 2020, LWPU CORPORATION.  All rights reserved.
 %
 % LWPU CORPORATION and its licensors retain all intellectual property
 % and proprietary rights in and to this software, related documentation
 % and any modifications thereto.  Any use, reproduction, disclosure or
 % distribution of this software and related documentation without an express
 % license agreement from LWPU CORPORATION is strictly prohibited.
 %%

function hdf5_dataset_to_fp16(infile, outfile, varargin)
  % HDF5_DATSET_TO_FP16 Colwert one or more datasets to fp16
  h5File = H5F.create(outfile, 'H5F_ACC_TRUNC', 'H5P_DEFAULT', 'H5P_DEFAULT');
  A = hdf5_load_lw(infile);
  names = fieldnames(A);
  for iName = 1:length(names)
      name = names{iName};
      for iColwertName = 1:numel(varargin)
          if strcmp(varargin{iColwertName}, name)
              fprintf('Colwerting %s to fp16\n', name);
              hdf5_write_lw(h5File, name, A.(name), 'fp16');
          else
              fprintf('Writing %s (unmodified)\n', name);
              hdf5_write_lw(h5File, name, A.(name));
          end   
      end
  end
  H5F.close(h5File);
