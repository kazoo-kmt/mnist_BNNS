//
//  BNNSCNNBasicViewController.swift
//  mnist_BNNS
//
//  Created by Kazu Komoto on 12/10/16.
//  Copyright © 2016 Kazu Komoto. All rights reserved.
//
/*
 This is based on MetalCNNBasicViewController.swift provided by Shuichi Tsutsumi.
 */

import UIKit
//import MetalPerformanceShaders

class BNNSCNNBasicViewController: UIViewController {

//    var commandQueue: MTLCommandQueue!
//    var device: MTLDevice!
    
    // Networks we have
    var network: MnistNet!
    
    // MNIST dataset image parameters
    let mnistInputWidth  = 28
    let mnistInputHeight = 28
    
    @IBOutlet private weak var digitView: DrawView!
    @IBOutlet private weak var predictionLabel: UILabel!
    @IBOutlet private weak var clearBtn: UIButton!
    
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        clearBtn.isHidden = true
        predictionLabel.text = nil

        // Load default device.
//        device = MTLCreateSystemDefaultDevice()
//        
//        // Make sure the current device supports MetalPerformanceShaders.
//        guard MPSSupportsMTLDevice(device) else {
//            showAlert(title: "Not Supported", message: "MetalPerformanceShaders is not supported on current device", handler: { (action) in
//                self.navigationController!.popViewController(animated: true)
//            })
//            return
//        }
//        
//        // Create new command queue.
//        commandQueue = device!.makeCommandQueue()
      
        // initialize the networks we shall use to detect digits
//        network  = MNISTDeepCNN(withCommandQueue: commandQueue)
        network = MnistNet()
    }
    
    @IBAction func clearBtnTapped(sender: UIButton) {
        // clear the digitview
        digitView.lines = []
        digitView.setNeedsDisplay()
        predictionLabel.text = nil
        clearBtn.isHidden = true
    }
    
    @IBAction func detectBtnTapped(sender: UIButton) {
        // get the digitView context so we can get the pixel values from it to intput to network
        let context = digitView.getViewContext()
        
        // validate NeuralNetwork was initialized properly
        assert(network != nil)
        
//        // putting input into MTLTexture in the MPSImage
//        network.srcImage.texture.replace(region: MTLRegion( origin: MTLOrigin(x: 0, y: 0, z: 0),
//                                                        size: MTLSize(width: mnistInputWidth, height: mnistInputHeight, depth: 1)),
//                                             mipmapLevel: 0,
//                                             slice: 0,
//                                             withBytes: context!.data!,
//                                             bytesPerRow: mnistInputWidth,
//                                             bytesPerImage: 0)

        // run the network forward pass
//        let imageData = Data(bytes: context!.data!, count: mnistInputWidth * mnistInputHeight * MemoryLayout<Float>.size)
        let imageData = Data(bytes: context!.data!, count: mnistInputWidth * mnistInputHeight)
        let label = network.predict(image: imageData)
        
        // show the prediction
        predictionLabel.text = "\(label)"
        clearBtn.isHidden = false
    }
}
