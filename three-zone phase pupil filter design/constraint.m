function [c,ceq]=constraint(x)
r0=0;
r3=1;
r1=x(1);
r2=x(2);
phase1=x(3);
phase2=x(4);
phase3=x(5);
I0=exp(i*phase1)*(r1.^2-r0.^2)+exp(i*phase2)*(r2.^2-r1.^2)+exp(i*phase3)*(r3.^2-r2.^2);
I1=0.5*(exp(i*phase1)*(r1.^4-r0.^4)+exp(i*phase2)*(r2.^4-r1.^4)+exp(i*phase3)*(r3.^4-r2.^4));
I2=1/3*(exp(i*phase1)*(r1.^6-r0.^6)+exp(i*phase2)*(r2.^6-r1.^6)+exp(i*phase3)*(r3.^6-r2.^6));
Uf=-2*imag(conj(I0)*I1)/(real(I0*conj(I2))-abs(I1).^2);
S=abs(I0).^2-0.5*Uf*imag(conj(I0)*I1);
ceq=0;
c=[Uf-1.5 -1.5-Uf 0.401-S r1-r2];
end