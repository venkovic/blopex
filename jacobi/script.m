font_size=24;
set(0, 'DefaultUIControlFontSize',font_size);

m1=menu('system size','8','20','100','200','1000','END');

if m1==6
   return;
end

if m1==1
   n=8;
elseif m1==2
   n=20;
elseif m1==3
   n=100;
elseif m1==4
   n=200;
elseif m1==5
   n=1000;
end % if-elseif
fprintf('matrix size %d\n',n);

m2=menu('block size','1','2','4','16','64','END');

if m2==6
   return
end
if m2==1
   k=1;
elseif m2==2
   k=2;
elseif m2==3
   k=4;
elseif m2==4
   k=16;
elseif m2==5
   k=64;
end % if-elseif
fprintf('block size %d\n',k);

% generate dense symmetric matrix of size n
A=randn(n,n); A=A+A';
fp=fopen('symmetric_matrix.mat','w');
fprintf(fp,'%d\n',n);
for i=1:n
    for j=1:n
        fprintf(fp,'%24.16e',A(i,j));
    end % for j
    fprintf(fp,'\n');
end % for i
fclose(fp);

[V,D,hist]=jacobi(A,k,eps);

figure(1)
clf
semilogy(hist,'+:b','LineWidth',2,'MarkerSize',12)
pause(0.1)
title('convergence history by sweeps');
xlabel('sweep')
ylabel('||off(A)||_F^2/||diag(A)||_F^2')
set(gca,'FontSize',font_size)
 

[VV,DD]=schur(A);
DD=spdiags(diag(DD),0,n,n);
% sort eigenvalues in decreasing order by magnitude
[~,I]=sort(abs(diag(DD)),'descend');
% permute eigenvectors and eigenvalues accordingly
VV=VV(:,I); DD=DD(I,I);

fprintf('relative error between Jacobi and QR eigenvalues %8.1e\n',...
	norm(diag(D)-diag(DD),inf)/norm(diag(DD),inf));
