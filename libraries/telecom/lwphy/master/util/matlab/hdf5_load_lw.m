 %%
 % Copyright (c) 2020, LWPU CORPORATION.  All rights reserved.
 %
 % LWPU CORPORATION and its licensors retain all intellectual property
 % and proprietary rights in and to this software, related documentation
 % and any modifications thereto.  Any use, reproduction, disclosure or
 % distribution of this software and related documentation without an express
 % license agreement from LWPU CORPORATION is strictly prohibited.
 %%


function s = hdf5_load_lw(filename)
  % HDF5_LOAD_LW() Load data from a lwPHY HDF5 file in MATLAB
  hdf5_struct = hdf5_get_lw(filename, '/');
  % If the user doesn't provide an output argument, load the variables
  % directly to the workspace
  if nargout < 1
      fields = fieldnames(hdf5_struct);
      for idx = 1:length(fields)
          assignin('base', fields{idx}, hdf5_struct.(fields{idx}));
      end
  else
      s = hdf5_struct;
  end
end

function s = hdf5_get_lw(filename, location)
    s = struct();
    info = h5info(filename, location);
    %- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    % Handle HDF5 groups
    for idx = 1:length(info.Groups)
        grp = info.Groups(idx);
        % Relwrsively load data
        s.(grp.Name) = hdf5_get_lw(filename, strcat(location, grp.Name, '/'));
    end
    %- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    % Handle HDF5 datasets
    for idx = 1:length(info.Datasets)
        dset = info.Datasets(idx);
        data = h5read(filename, strcat(location, dset.Name)); 
        % HDF5 doesn't have native support for complex data types. The
        % hdf5_write_lw() function maps MATLAB complex data to an HDF5
        % compound data type with members called 're' and 'im'. If we
        % find those, manually create a MATLAB complex data type.
        if (strcmp(dset.Datatype(1).Class, 'H5T_COMPOUND'))     && ...
           (length(dset.Datatype(1).Type.Member) == 2)          && ...
           (strcmp(dset.Datatype(1).Type.Member(1).Name, 're')) && ...
           (strcmp(dset.Datatype(1).Type.Member(2).Name, 'im'))
            s.(dset.Name) = data.re + (i * data.im);
            %fprintf('%s is complex\n', dset.Name);
        elseif strcmp(dset.Datatype(1).Class, 'H5T_COMPOUND')
            % Colwert from structure of arrays back to array of structs
            names = fieldnames(data);
            aos_sz = [1 1];
            aos = struct();
            for iName = 1:length(names)
                values = data.(names{iName});
                % (All sizes should be the same...)
                aos_sz = size(values);
                % flatten so that we can assign
                values_flat = values(:);
                % Assign a value to the field for each element
                for idx = 1:numel(values_flat)
                    aos(idx).(names{iName}) = values_flat(idx);
                end
            end
            % Reshape to the original dimensions
            s.(dset.Name) = reshape(aos, aos_sz);
        else
            % For non-complex data, just store what was read...
            s.(dset.Name) = data;
        end
    end
end
