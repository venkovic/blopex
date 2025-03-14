function [V,D,hist]=jacobi(A,block_size,tol)
% [V,D,hist]=jacobi(A,block_size,tol)
%
% Jacobi-like method for computing eigenvalues and eigenvectors of a symmetric
% matrix
%
% Input
% -----
% A           symmetric matrix
% block_size  block size, Jacobi method solves cyclically sub problems of size
%             2*block_size x 2*block_size
% tol         termination threshold. Jacobi method stops whenever
%             sum_{i~=j} A(i,j)^2 <= tol * sum_{i} A(i,i)^2
%
% Output
% ------
% V     matrix of (orthogonal) eigenvectors
% D     diagonal matrix with real eigenvalues

% size of the input matrix  
n=size(A,1);
% store size in case we have to enlarge it  
nn=n;

% check whether the block size fits into the matrix
l=mod(nn,2*block_size);
% block size does not exactly fit, add zero rows and columns
if l
   m=2*block_size-l;
   A=[    A       zeros(nn,m);...
      zeros(m,nn) zeros(m,m)];
   n=size(A,1)
   % now n is a mutliple of 2*block_size
end
% number of parallel problems to be solved simultaneously
m=n/(2*block_size);
% initial partitioning (1,2), (3,4), (5,6), ... (2m-1,2m)
coord=reshape(1:2*m,2,m)';

% initial diagonal matrix = input matrix
D=A;
% initial eigenvector matrix = I
V=eye(n);

% square sum of the diagonal entries
nrm_dgl=norm(diag(D),2)^2;
% square sum of the off-diagonal entries
nrm_offdgl=norm(D,'fro')^2-nrm_dgl;

iter=0;
while nrm_offdgl>tol*nrm_dgl
      iter=iter+1;
      hist(iter)=nrm_offdgl/nrm_dgl;
      
      % compute yet another sweep
      for i=1:2*m-1
	  % this loop could in prinple be performed in parallel
	  % but multiplication of D must be synchronized, at first
	  % multiplication from the right with VV is done in common
	  % after that, when all multiplications D(:,[I J])*VV are
	  % finished, then all multiplications VV'*D([I J],:) can
	  % be computed
	
	  % first stage: solve small eigenvalue problem and update norms
	  for j=1:m
	      % extract block number (r,s)
	      r=coord(j,1); s=coord(j,2);
	      % solve eigenvalue problem of block (r,s)
	      I=(r-1)*block_size+1:r*block_size; % first sub block
	      J=(s-1)*block_size+1:s*block_size; % second sub block
	      % local symmetric matrix block
	      AA=D([I J],[I J]);

	      % local square sum of the diagonal entries
	      nrm_dgl_loc   =norm(diag(AA),2)^2;
              % local square sum of the off-diagonal entries
	      nrm_offdgl_loc=norm(AA,'fro')^2-nrm_dgl_loc;
	      % downdate squared norm of the diagonal entries
	      nrm_dgl   =nrm_dgl-nrm_dgl_loc;
	      % downdate squared norm of the off-diagonal entries
	      nrm_offdgl=nrm_offdgl-nrm_offdgl_loc;

	      % solve local eigenvalue problem
	      [VV{j},DD{j}]=schur(AA);
	      % find out which diagonal entries of DD are closest
	      % to the diagonal entries of AA. This is necessary
	      % to make sure that VV tends to I when AA tends to diagonal
	      [~,permA]=sort(diag(AA));
	      [~,permD]=sort(diag(DD{j}));
	      VV{j}=VV{j}(:,permD); DD{j}=DD{j}(permD,permD);
	      VV{j}(:,permA)=VV{j}; DD{j}(permA,permA)=DD{j};
	      
	      % update squared norm of the diagonal entries
	      nrm_dgl=nrm_dgl+norm(diag(DD{j}),2)^2;
	  end % for j

	  % second stage: multiplication from the right
	  for j=1:m
	      % extract block number (r,s)
	      r=coord(j,1); s=coord(j,2);
	      % solve eigenvalue problem of block (r,s)
	      I=(r-1)*block_size+1:r*block_size; % first sub block
	      J=(s-1)*block_size+1:s*block_size; % second sub block
	      % update eigenvector matrix
	      V(:,[I J])=V(:,[I J])*VV{j};
	      % update eigenvalue matrix
	      D(:,[I J])=D(:,[I J])*VV{j};
	  end % for j

	  % third stage: multiplication of D from the left
	  for j=1:m
	      % extract block number (r,s)
	      r=coord(j,1); s=coord(j,2);
	      % solve eigenvalue problem of block (r,s)
	      I=(r-1)*block_size+1:r*block_size; % first sub block
	      J=(s-1)*block_size+1:s*block_size; % second sub block
	      % update eigenvalue matrix
	      D([I J],:)=VV{j}'*D([I J],:);
	  end % for j
	  
	  % fourth stage: diagonal blocks and new coordinates
	  for j=1:m
	      % extract block number (r,s)
	      r=coord(j,1); s=coord(j,2);
	      % solve eigenvalue problem of block (r,s)
	      I=(r-1)*block_size+1:r*block_size; % first sub block
	      J=(s-1)*block_size+1:s*block_size; % second sub block
	      % update eigenvalue matrix
	      D([I J],[I J])=DD{j};
	      
	      % new r
	      if r~=1
		 % even case
		 if mod(r,2)==0
		    % increase by 2 if possible
		    if r<2*m
		       r=r+2;
		    else % => r=2*m
		       r=2*m-1;
		    end % if-else
		 else % odd case
		    % decrease by 2 if possible
		    if r>3
		       r=r-2;
		    else % => r=3
		       r=2;
		    end % if-else
		 end % if-else 
	      end

	      % new s, same procedure
	      if s~=1
		 % even case
		 if mod(s,2)==0
		    % increase by 2 if possible
		    if s<2*m
		       s=s+2;
		    else % => s=2*m
		       s=2*m-1;
		    end % if-else
		 else % odd case
		    % decrease by 2 if possible
		    if s>3
		       s=s-2;
		    else % => s=3
		       s=2;
		    end % if-else
		 end % if-else 
	      end

	      % coordinate for next sweep
	      coord(j,1)=r; coord(j,2)=s;
	  end % for j
      end % for i      
end % while

% for clearness, remove off-diagonal entries, extract diagonal entries as sparse
D=spdiags(diag(D),0,n,n);
% sort eigenvalues in decreasing order by magnitude
%[~,I]=sort(abs(diag(D)),'descend');
[~,I]=sort(diag(D));
% permute eigenvectors and eigenvalues accordingly
V=V(:,I); D=D(I,I);

% remove additional entries
l=mod(nn,2*block_size);
if l
   m=2*block_size-l;
   V=V(1:nn,:);
   r=0;
   for j=n:-1:1
       if norm(V(:,j))<=1e-1
          V(:,j)=[];
	  D(:,j)=[];
	  D(j,:)=[];
	  r=r+1;
	  if r==m
	     break;
	  end
       end % if
   end % for j
end % if


  
end % function
