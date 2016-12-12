//
//  MnistNet.swift
//  mnist_BNNS
//
//  Created by Kazu Komoto on 11/25/16.
//  Copyright © 2016 Kazu Komoto. All rights reserved.
//
//
/*
 This is based on MnistNet.swift provided by Pavel Ivashkov.
 */


import Foundation
import UIKit


class MnistNet {
    
    init() {
        mnistNetwork = MnistNet.setupNetwork()
    }
  
    let mnistNetwork: BnnsNetwork

  enum ParamsChoice: String {
    case weights
    case bias
  }
  
  private class func readParams(kernelWidth: UInt, kernelHeight: UInt, inputFeatureChannels: UInt, outputFeatureChannels: UInt, kernelParamsBinaryName: String) -> Array<Array<Float32>> {
//  private class func readParams(kernelWidth: UInt, kernelHeight: UInt, inputFeatureChannels: UInt, outputFeatureChannels: UInt, kernelParamsBinaryName: String) -> [UnsafePointer<Float32>] {

      // calculate the size of weights and bias required to be memory mapped into memory
      let sizeWeights = inputFeatureChannels * kernelHeight * kernelWidth * outputFeatureChannels * UInt(MemoryLayout<Float>.size)
      let sizeBias = outputFeatureChannels * UInt(MemoryLayout<Float>.size)

      // get the url to this layer's weights and bias
      let wtPath = Bundle.main.path(forResource: "weights_" + kernelParamsBinaryName, ofType: "dat")
      let bsPath = Bundle.main.path(forResource: "bias_" + kernelParamsBinaryName, ofType: "dat")

    
      // open file descriptors in read-only mode to parameter files
      let fd_w = open(wtPath!, O_RDONLY, S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP | S_IROTH | S_IWOTH)
      let fd_b = open(bsPath!, O_RDONLY, S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP | S_IROTH | S_IWOTH)
    
      assert(fd_w != -1, "Error: failed to open output file at \""+wtPath!+"\"  errno = \(errno)\n")
      assert(fd_b != -1, "Error: failed to open output file at \""+bsPath!+"\"  errno = \(errno)\n")
    
      // memory map the parameters
      let hdrW = mmap(nil, Int(sizeWeights), PROT_READ, MAP_FILE | MAP_SHARED, fd_w, 0)
      let hdrB = mmap(nil, Int(sizeBias), PROT_READ, MAP_FILE | MAP_SHARED, fd_b, 0)
    
      // cast Void pointers to Float
      let w = UnsafePointer<Float>(hdrW!.assumingMemoryBound(to: Float.self))
      let b = UnsafePointer<Float>(hdrB!.assumingMemoryBound(to: Float.self))
    
      assert(w != UnsafePointer<Float>(bitPattern: -1), "mmap failed with errno = \(errno)")
      assert(b != UnsafePointer<Float>(bitPattern: -1), "mmap failed with errno = \(errno)")
    
      let bufferPointer_w = UnsafeBufferPointer(start: w, count: Int(sizeWeights/UInt(MemoryLayout<Float>.size)))
      let bufferPointer_b = UnsafeBufferPointer(start: b, count: Int(sizeBias/UInt(MemoryLayout<Float>.size)))
      let array_w = Array(bufferPointer_w)
      let array_b = Array(bufferPointer_b)
    
      // unmap files at initialization of MPSCNNConvolution, the weights are copied and packed internally we no longer require these
      assert(munmap(hdrW, Int(sizeWeights)) == 0, "munmap failed with errno = \(errno)")
      assert(munmap(hdrB, Int(sizeBias))    == 0, "munmap failed with errno = \(errno)")
      
      // close file descriptors
      close(fd_w)
      close(fd_b)
    
//      return params
      return [array_w, array_b]
    }
  
    private class func setupNetwork() -> BnnsNetwork {
      
        let conv1_params = readParams(kernelWidth: 3, kernelHeight: 3, inputFeatureChannels: 1, outputFeatureChannels: 32, kernelParamsBinaryName: "conv1")
        let conv2_params = readParams(kernelWidth: 3, kernelHeight: 3, inputFeatureChannels: 32, outputFeatureChannels: 32, kernelParamsBinaryName: "conv2")
        let fc1_params = readParams(kernelWidth: 12, kernelHeight: 12, inputFeatureChannels: 32, outputFeatureChannels: 128, kernelParamsBinaryName: "fc1")
        let fc2_params = readParams(kernelWidth: 1, kernelHeight: 1, inputFeatureChannels: 128, outputFeatureChannels: 10, kernelParamsBinaryName: "fc2")
      
        return BnnsBuilder()
            .shape(width: 28, height: 28, channels: 1)
            .kernel(width: 3, height: 3)
            .activation(function: BNNSActivationFunctionRectifiedLinear)
            .convolve(weights: conv1_params[0], bias: conv1_params[1])
          
            .shape(width: 26, height: 26, channels: 32)
            .convolve(weights: conv2_params[0], bias: conv2_params[1])
          
            .shape(width: 24, height: 24, channels: 32)
            .maxpool(width: 2, height: 2)
              
            .shape(width: 12, height: 12, channels: 32)
            .kernel(width: 12, height: 12)
            .connect(weights: fc1_params[0], bias: fc1_params[1])

            .shape(size: 128)
//            .shape(width: 1, height: 1, channels: 128)
            .activation(function: BNNSActivationFunctionSigmoid)  // Note: It should be softmax, but used sigmoid because BNNS hasn't provided softmax yet.
            .connect(weights: fc2_params[0], bias: fc2_params[1])
          
            .shape(size: 10)
//            .shape(width: 1, height: 1, channels: 10)
          /*
            .shape(width: 28, height: 28, channels: 1)
            .kernel(width: 5, height: 5)
            .convolve(weights: conv1_params[0], bias: conv1_params[1])
            .shape(width: 28, height: 28, channels: 32)
            .maxpool(width: 2, height: 2)
            .shape(width: 14, height: 14, channels: 32)
            .convolve(weights: conv2_params[0], bias: conv2_params[1])
            .shape(width: 14, height: 14, channels: 64)
            .maxpool(width: 2, height: 2)
            .shape(width: 7, height: 7, channels: 64)
            .connect(weights: fc1_params[0], bias: fc1_params[1])
            .shape(size: 1024)
            .connect(weights: fc2_params[0], bias: fc2_params[1])
            .shape(size: 10)
          */
            .build()!
    }
    
    func predict(image: Data) -> Int {
        return predict(input: read(image: image))
    }
    
    func predict(input: [Float32]) -> Int {
        
        let outputs = mnistNetwork.apply(input: input)

        return outputs.index(of: outputs.max()!)!
    }
    
    func predictBatch(images: Data, count: Int) -> [Int] {
        
        let outputs = mnistNetwork
            .batch(input: read(image: images), count: count)
            .map { $0.index(of: $0.max()!)! }

        return outputs
    }

    private func read(image: Data) -> [Float32] {
        return image.map { Float32($0) / 255.0 }
    }
}
