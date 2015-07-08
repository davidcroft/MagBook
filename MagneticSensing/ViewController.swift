//
//  ViewController.swift
//  MagneticSensing
//
//  Created by Ding Xu on 12/13/14.
//  Copyright (c) 2014 Ding Xu. All rights reserved.
//

import UIKit
import CoreMotion

class ViewController: UIViewController {

    @IBOutlet var magValX: UILabel!
    @IBOutlet var magValY: UILabel!
    @IBOutlet var magValZ: UILabel!
    @IBOutlet var mlStatus: UILabel!
    @IBOutlet var captureButton: UIButton!
    
    var motionManager: CMMotionManager = CMMotionManager()
    var magnetoTimer: NSTimer!
    
    var isTrainedFlag: Bool = false
    var verboseFlag: Int = 0   // 0: init, hidden buttons;  1: train; 2: predict
    var startCapFlag: Bool = false // flag for storing data into list
    
    var trainingSampleCnt:Int = 0
    let TRAINSAMPLENUM:Int = 50
    let TRAINCLASSNUM:Int = 4
    
    //var trainingData = [Double, Double, Double]()
    //var classTag:[Double] = []
    var trainingData:ndarray
    var trainingTag:ndarray
    var classIndex:Int = 0
    var svm = SVM()

    required init(coder aDecoder: NSCoder) {
        // 3 axis val of magneto meter
        self.trainingData=ndarray(n: 3*self.TRAINCLASSNUM * self.TRAINSAMPLENUM)
        // tag for training data
        self.trainingTag=ndarray(n: self.TRAINCLASSNUM * self.TRAINSAMPLENUM)
        super.init(coder: aDecoder)
    }

    
    override func viewDidLoad() {
        super.viewDidLoad()
        // Do any additional setup after loading the view, typically from a nib.
        
        // TODO
        self.magValX.text = "X: 0"
        self.magValY.text = "Y: 0"
        self.magValZ.text = "Z: 0"
        
        self.motionManager.startMagnetometerUpdates()
        
        // add a timer and new thread to get data from sensors
        self.magnetoTimer = NSTimer.scheduledTimerWithTimeInterval(0.2,
            target:self,
            selector:"updateMegneto:",
            userInfo:nil,
            repeats:true)
        
        // machine learning init
        self.updateVerbose()
    }
    
    func updateVerbose() {
        if(verboseFlag == 0) {
            // init
            self.mlStatus.hidden = true
            self.captureButton.hidden = true
        } else if (verboseFlag == 1){
            // train
            self.mlStatus.hidden = false
            self.captureButton.hidden = false
        } else if (verboseFlag == 2) {
            // predict
            self.mlStatus.hidden = false
            self.captureButton.hidden = true
        }
    }

    override func didReceiveMemoryWarning() {
        super.didReceiveMemoryWarning()
        // Dispose of any resources that can be recreated.
    }
    
    func updateMegneto(timer: NSTimer) -> Void {
        // TODO
        dispatch_async(dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), { () -> Void in
            if self.motionManager.magnetometerData != nil {
                let valX = self.motionManager.magnetometerData.magneticField.x
                let valY = self.motionManager.magnetometerData.magneticField.y
                let valZ = self.motionManager.magnetometerData.magneticField.z
                // update text in main thread
                dispatch_async(dispatch_get_main_queue(), { () -> Void in
                    self.magValX.text = String(format:"X: %f", valX)
                    self.magValY.text = String(format:"Y: %f", valY)
                    self.magValZ.text = String(format:"Z: %f", valZ)
                })
                
                if (self.verboseFlag == 1 && self.startCapFlag) {
                    // capture data
                    if self.trainingSampleCnt < self.TRAINSAMPLENUM {
                        let sampleIndex = self.trainingSampleCnt + (self.classIndex-1)*self.TRAINSAMPLENUM
                        self.trainingData[3*sampleIndex] = valX
                        self.trainingData[3*sampleIndex+1] = valY
                        self.trainingData[3*sampleIndex+2] = valZ
                        //self.trainingData.append(valX, valY, valZ)
                        println("\(sampleIndex) - X:\(self.trainingData[3*sampleIndex]), Y:\(self.trainingData[3*sampleIndex]), Z:\(self.trainingData[3*sampleIndex])")
                        //self.classTag.append(Double(self.classIndex))
                        self.trainingTag[sampleIndex] = Double(self.classIndex)
                        self.trainingSampleCnt++
                        dispatch_async(dispatch_get_main_queue(), { () -> Void in
                            self.captureButton.enabled = false
                        })
                    } else {
                        // stop capture
                        self.startCapFlag = false
                        // set text
                        dispatch_async(dispatch_get_main_queue(), { () -> Void in
                            if self.classIndex >= self.TRAINCLASSNUM {
                                self.mlStatus.text = "all data capture finished"
                                self.captureButton.setTitle("Start training", forState: UIControlState.Normal)
                            } else {
                                self.mlStatus.text = "capture \(self.classIndex) finished, ready for next"
                            }
                            self.captureButton.enabled = true
                        })
                        println("")
                    }
                } else if (self.verboseFlag == 2) {
                    var vals = ones(3)
                    vals[0] = valX
                    vals[1] = valY
                    vals[2] = valZ
                    var res = self.svm.predict(vals)
                    dispatch_async(dispatch_get_main_queue(), { () -> Void in
                        self.mlStatus.text = "Position class: \(res)"
                    })
                }
            }
        })
    }

    @IBAction func capTrainingDataStart(sender: AnyObject) {
        // check if trained
        if (isTrainedFlag) {
            self.mlStatus.text = "model has been trained!"
        } else {
            if(self.classIndex < self.TRAINCLASSNUM) {
                self.classIndex++
                // capture data init
                self.startCapFlag = true
                self.trainingSampleCnt = 0
                self.mlStatus.text = "capture class \(self.classIndex) data"
            } else {
                self.isTrainedFlag = true
                // trainModel
                var responses = reshape(self.trainingData, (self.TRAINCLASSNUM * self.TRAINSAMPLENUM, 3))
                svm.train(responses, self.trainingTag)
                // status
                self.mlStatus.text = "Training finished!"
                self.captureButton.hidden = true
                verboseFlag = 2
            }
        }
    }

    @IBAction func trainBtn(sender: AnyObject) {
        // train
        self.verboseFlag = 1
        self.updateVerbose()
        //
        if (isTrainedFlag) {
            self.mlStatus.text = "model has been trained!"
            self.captureButton.hidden = true
        } else {
            self.mlStatus.text = "capture class \(self.classIndex+1) data"
        }
    }

    @IBAction func predictBtn(sender: AnyObject) {
        // predict
        if (!isTrainedFlag) {
            self.mlStatus.hidden = false
            self.mlStatus.text = "please train model first"
            return
        }
        self.verboseFlag = 2
        self.updateVerbose()
    }
    
    @IBAction func saveBtn(sender: AnyObject) {
        self.svm.save("/svmModel.xml")
    }
    
    @IBAction func loadBtn(sender: AnyObject) {
        self.svm.load("/svmModel.xml")
    }
}

