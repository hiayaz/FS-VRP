
clc; clear;
nJ=[5 10 20 40 80];
nM=[5 10 15 20];
nG=[10 20 30];
folder = 'C:\Users\User\OneDrive - erbakan.edu.tr\esra\Matlab\Chat'; 
for i1=1:numel(nM)
    for i2=1:numel(nJ)
        for i3=1:numel(nG) 
            nJobs=nJ(i2);
            mMachines=nM(i1);            
            gridSize=nG(i3);
            filename = sprintf('Machine%d--Job%d--Grid%d.mat', mMachines, nJobs, gridSize);
            fullpath = fullfile(folder, filename);
            load(fullpath);
            parfor i4=1:1
                compareMetaheuristics(problem,nJobs,mMachines,gridSize,i4)
            end
        end
    end
end