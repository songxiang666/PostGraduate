n=26;
x=[];
for i=1:n
    x1=find(isolet(:,618)==i, 1, 'last' );
    x=[x,x1];
end
y=[];
for i=1:n-1
    y=[y,x(i+1)-x(i)];
end
y=[x(1),y];
z=round(y.*0.5);
s=[];
s=[s,1];
for i=1:n-1
    s=[s,x(i),x(i)+1];
end
s=[s,x(n)];