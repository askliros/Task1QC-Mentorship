# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 12:26:53 2021

@author: aris_
"""


import numpy as np
#from BasicGates import *
#from SwapClass import *
import math
import os
os.getcwd()
import copy

class SWAPGate:
    def __init__(self,qubit1,qubit2):
        self.qubit1=qubit1
        self.qubit2=qubit2
    def applyGate(self):
        self.ComposedQubit=np.kron(self.qubit1,self.qubit2)
        self.OperationMatrix=np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])
        self.OperationMatrixOrig=self.OperationMatrix.copy()
        self.Result=np.matmul(self.OperationMatrix,self.ComposedQubit)
    def reset(self):
        self.OperationMatrix=self.OperationMatrixOrig
    def __mul__(self,other):
        NewObject=type(self)
        NewObject.OperationMatrix=(np.matmul(self.OperationMatrix,other.OperationMatrix))
        return NewObject
    def __add__(self,other):
        NewObject=type(self)
        NewObject.OperationMatrix=(np.kron(self.OperationMatrix,other.OperationMatrix))
        return NewObject

class Rmatrix:
    def __init__(self,theta):
        self.theta=theta
    def applyGate(self):
        self.OperationMatrix=np.array([[np.cos(self.theta/2.0),-np.sin(self.theta/2.0)],[np.sin(self.theta/2.0),np.cos(self.theta/2.0)]])
class Rz:
    def __init__(self,theta):
        self.theta=theta
    def applyGate(self):
        self.OperationMatrix=np.array([[1,0],[0,np.exp(1j*self.theta)]])

class HadamardGate:
    def __init__(self,qubit):
        self.qubit=qubit
    def applyGate(self):
        self.ComposedQubit=self.qubit
        self.OperationMatrix=((1.0)/np.sqrt(2.0))*np.array([[1,1],[1,-1]])
        self.OperationMatrixOrig=((1.0)/np.sqrt(2.0))*np.array([[1,1],[1,-1]])
        self.Result=np.matmul(self.OperationMatrix,self.ComposedQubit)
    def reset(self):
        self.OperationMatrix=self.OperationMatrixOrig
    def __mul__(self,other):
        NewObject=type(self)
        NewObject.OperationMatrix=(np.matmul(self.OperationMatrix,other.OperationMatrix))
        return NewObject
    def __add__(self,other):
        NewObject=type(self)
        NewObject.OperationMatrix=(np.kron(self.OperationMatrix,other.OperationMatrix))
        return NewObject

class Variational:
    def __init__(self,theta,phi):
        self.theta=theta
        self.phi=phi
    def applyGate(self):
        self.Rotation=Rmatrix(self.theta)
        self.Rotation.applyGate()
        self.R=self.Rotation.OperationMatrix
        self.PhaseShift=Rz(self.phi)
        self.PhaseShift.applyGate()
        self.Phase=self.PhaseShift.OperationMatrix
        self.OperationMatrix=np.matmul(self.Phase,self.R)

class SWAPGateCircuit:
    def __init__(self):
        pass
    def applyGate(self):
        qubit=np.array([1,0])
        qubit2=np.array([1,0])
        MyHadamard=HadamardGate(qubit)
        MyHadamard.applyGate()
        self.H=MyHadamard.OperationMatrix
        self.RightPart=np.kron(self.H,np.kron(np.eye(2),np.eye(2)))
        self.LeftPart=np.kron(self.H,np.kron(np.eye(2),np.eye(2)))
        MySwap=SWAPGate(qubit,qubit2)
        MySwap.applyGate()
        self.Swap=MySwap.OperationMatrix
        self.ControlledSwap=np.eye(8)
        self.ControlledSwap[4:8,4:8]=self.Swap
        self.OperationMatrix=np.matmul(self.LeftPart,np.matmul(self.ControlledSwap,self.RightPart))
        
class SwapTest:
    def __init__(self,AncillaQubit,PsiQubit,PhiQubit):
        self.AncillaQubit=AncillaQubit
        self.PsiQubit=PsiQubit
        self.PhiQubit=PhiQubit
        self.SwapGate=SWAPGateCircuit()
    def applyGate(self):
        self.SwapGate.applyGate()
        self.Circuit=self.SwapGate.OperationMatrix
        self.Output=np.matmul(self.Circuit,np.kron(self.AncillaQubit,np.kron(self.PsiQubit,self.PhiQubit)))
class Simulation:
    def __init__(self):
        self.AncillaQubit=np.array([1,0])
        
    def generatePsiQubit(self):
        #Generate a Quantum State randomly:
        #Defining the Qubit Psi randomly:
        # RandomNumber=np.random.randint(0,2)
        # if(RandomNumber==1):
        #     self.PsiQubit=np.array([0,1])
        # elif(RandomNumber==0):
        #     self.PsiQubit=np.array([1,0])
        # else:
        #     self.PsiQubit=np.array([1,0]) 
        #The qubit Psi is the randomly generated quantum state
        theta0=np.random.uniform(0,math.pi)
        phi0=np.random.uniform(0,2*math.pi)
        Var0=Variational(theta0,phi0)
        Var0.applyGate()
        #Finding the Qubit Psi
        self.PsiQubit=np.matmul(Var0.OperationMatrix,np.array([1,0]))
        print("The randomly generated PsiQubit is:")
        print(self.PsiQubit)         
        self.applySimulation()
        
    def applySimulation(self): 
        #Now we need to find the Qubit Phi which reproduces the Qubit Psi
        # we perform exhaustive test until Pr_0=1        
        breaker=False
        NumSimulations=100
        self.theta=0
        self.phi=0
        for thetak in range(0,NumSimulations):
            for phik in range(0,NumSimulations):
                theta=(1.0/float(NumSimulations))*thetak*math.pi
                phi=(1.0/float(NumSimulations))*phik*2*math.pi
                Var=Variational(theta,phi)
                Var.applyGate()
                #Finding the Qubit Phi
                self.PhiQubit=np.matmul(Var.OperationMatrix,np.array([1,0]))
                #self.PsiQubit=np.array([0,1])
                #self.PhiQubit=np.array([1,0])
                self.Test=SwapTest(self.AncillaQubit,self.PsiQubit,self.PhiQubit)
                self.Test.applyGate()
                self.Result=self.Test.Output
                self.Pr_0=np.dot(Sim.Result[:4].conjugate(),Sim.Result[:4])
                if(np.abs(self.Pr_0-1.0)<1e-4):
                    print("Yes")
                    breaker=True
                    break
            if(breaker):
                self.theta=theta
                self.phi=phi
                break
        print("The approximated PhiQubit is:")
        print(self.PhiQubit)    
            
    def generateNQubitState(self,N):
        self.NQubitPsiState=[]
        self.NQubitPhiState=[]
        self.NQubitPsi=np.array([1,0])
        self.NQubitPhi=np.array([1,0])
        for kkk in range(0,N): 
            theta0=np.random.uniform(0,math.pi)
            phi0=np.random.uniform(0,2*math.pi)
            Var0=Variational(theta0,phi0)
            Var0.applyGate()
            #Finding the Qubit Psi
            self.PsiQubit=np.matmul(Var0.OperationMatrix,np.array([1,0]))
            self.NQubitPsiState.insert(len(self.NQubitPsiState),self.PsiQubit)            
            print("The randomly generated PsiQubit is:")
            print(self.PsiQubit) 
            #We now approximate the PhiQubit and we insert it in the list
            self.applySimulation()
            self.NQubitPhiState.insert(len(self.NQubitPhiState),self.PhiQubit)
            if(kkk==0):
                self.NQubitPsi=np.copy(self.PsiQubit)
                self.NQubitPhi=np.copy(self.PhiQubit)                
            else:
                self.NQubitPsi=np.kron(self.NQubitPsi,self.PsiQubit)
                self.NQubitPhi=np.kron(self.NQubitPhi,self.PhiQubit)
                
                
            
            
            
            
            
        
        
        
Sim=Simulation()
Sim.generatePsiQubit()
Sim.Result  
#Sim.generateNQubitState(10)     
        



        
# theta=math.pi/6
# phi=math.pi/7
# Var=Variational(theta,phi)
# Var.applyGate()
# Var.OperationMatrix

# SwapGate=SWAPGateCircuit()
# SwapGate.applyGate()
# print(SwapGate.OperationMatrix)

# for kkk in range(0,100):
#     print(np.random.randint(0,2))