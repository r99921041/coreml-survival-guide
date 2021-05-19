import CoreMedia
import CoreML
import UIKit
import Vision

import MobileCoreServices
import AVFoundation

class ViewController: UIViewController {

  @IBOutlet var videoPreview: UIView!

  var videoCapture: VideoCapture!
  var currentBuffer: CVPixelBuffer?

  let coreMLModel = MobileNetV2_SSDLite()

  lazy var visionModel: VNCoreMLModel = {
    do {
      return try VNCoreMLModel(for: coreMLModel.model)
    } catch {
      fatalError("Failed to create VNCoreMLModel: \(error)")
    }
  }()

  lazy var visionRequest: VNCoreMLRequest = {
    let request = VNCoreMLRequest(model: visionModel, completionHandler: {
      [weak self] request, error in
      self?.processObservations(for: request, error: error)
    })

    // NOTE: If you use another crop/scale option, you must also change
    // how the BoundingBoxView objects get scaled when they are drawn.
    // Currently they assume the full input image is used.
    request.imageCropAndScaleOption = .scaleFill
    return request
  }()

  let maxBoundingBoxViews = 10
  var boundingBoxViews = [BoundingBoxView]()
  var colors: [String: UIColor] = [:]

  fileprivate var frameIndex = -1
  fileprivate var indexFirstDetectedPerson: Int?
  fileprivate var indexLastDetectedPerson: Int?

  fileprivate var firstVideoTrack: AVAssetTrack?
  fileprivate var firstSampleToRecordTime = CMTime.zero

  fileprivate var writeHelper: (writer: AVAssetWriter, input: AVAssetWriterInput)? {
    if writeHelperBackingStore == nil {
      writeHelperBackingStore = makeAssetWriter(
        source: firstVideoTrack,
        sourceTime: firstSampleToRecordTime)
    }
    return writeHelperBackingStore
  }

  fileprivate var writeHelperBackingStore: (writer: AVAssetWriter, input: AVAssetWriterInput)?

  fileprivate lazy var loadingIndicator = UIView()

  override func viewDidLoad() {
    super.viewDidLoad()
    setUpBoundingBoxViews()
    setUpCamera()
    setupLoadingIndicator()
    showVideoPicker()
  }

  fileprivate func setupLoadingIndicator() {
    guard let view = view else {
      return
    }
    loadingIndicator.backgroundColor = UIColor.black.withAlphaComponent(0.75)
    loadingIndicator.layer.cornerRadius = 20
    loadingIndicator.layer.masksToBounds = true
    let indicator = UIActivityIndicatorView(style: .whiteLarge)
    indicator.startAnimating()

    loadingIndicator.addSubview(indicator)
    indicator.translatesAutoresizingMaskIntoConstraints = false
    indicator.centerXAnchor.constraint(equalTo: loadingIndicator.centerXAnchor).isActive = true
    indicator.centerYAnchor.constraint(equalTo: loadingIndicator.centerYAnchor).isActive = true

    view.addSubview(loadingIndicator)
    loadingIndicator.translatesAutoresizingMaskIntoConstraints = false
    loadingIndicator.centerXAnchor.constraint(equalTo: view.centerXAnchor).isActive = true
    loadingIndicator.centerYAnchor.constraint(equalTo: view.centerYAnchor).isActive = true
    loadingIndicator.widthAnchor.constraint(equalToConstant: 80).isActive = true
    loadingIndicator.heightAnchor.constraint(equalToConstant: 80).isActive = true

    loadingIndicator.isHidden = true
  }

  fileprivate func showVideoPicker() {
    let imagePickerVC = UIImagePickerController()
    imagePickerVC.mediaTypes = [kUTTypeMovie] as [String]
    imagePickerVC.delegate = self
    present(imagePickerVC, animated: true)
  }
}

extension ViewController: UIImagePickerControllerDelegate, UINavigationControllerDelegate {

  func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey : Any]) {
    if let fileURL = info[.mediaURL] as? URL {
      loadingIndicator.isHidden = false
      DispatchQueue.global(qos: .userInitiated).async { [weak self] in
        self?.loadAndProcess(fileURL)
      }
    }
    let presentingVC = picker.presentingViewController ?? self
    presentingVC.dismiss(animated: true)
  }
}

extension ViewController {

  fileprivate func loadAndProcess(_ fileURL: URL) {
    let asset = AVAsset(url: fileURL)
    guard let reader = try? AVAssetReader(asset: asset) else {
      return
    }
    let videoTracks = asset.tracks(withMediaType: .video)
    firstVideoTrack = videoTracks.first
    let output = AVAssetReaderVideoCompositionOutput(videoTracks: videoTracks, videoSettings: nil)
    output.videoComposition = AVVideoComposition(propertiesOf: asset)
    if reader.canAdd(output) {
      reader.add(output)
    }
    reader.startReading()
    while reader.status == .reading,
          let sample = output.copyNextSampleBuffer() {
      frameIndex += 1
      predict(sampleBuffer: sample)
      let shouldFinishWriting = writeIfNeeded(sample)
      if shouldFinishWriting {
        finishWritingIfNeeded()
      }
    }
    finishWritingIfNeeded()
    if reader.status == .reading {
      reader.cancelReading()
    }
    print("Processing input video done.\nreader.status \(reader.status.rawValue)\nLast frame index \(frameIndex)")
    DispatchQueue.main.async { [weak self] in
      self?.loadingIndicator.isHidden = true
    }
  }

  fileprivate func makeAssetWriter(
    source track: AVAssetTrack?,
    sourceTime: CMTime
  ) -> (
    writer: AVAssetWriter,
    input: AVAssetWriterInput
  )?
  {
    guard let track = track else {
      return nil
    }
    let fileManager = FileManager.default
    let documentFolderURL = (try? fileManager.url(
      for: .documentDirectory,
      in: .userDomainMask,
      appropriateFor: nil,
      create: true)) ??
      URL(fileURLWithPath: NSTemporaryDirectory())
    let dateFormatter = DateFormatter()
    dateFormatter.dateFormat = "yyyy-MM-dd-HH-mm-ss"
    let dateNow = Date(timeIntervalSinceNow: 0)
    let fileName = dateFormatter.string(from: dateNow)
    let outputFileURL = documentFolderURL
      .appendingPathComponent(fileName)
      .appendingPathExtension("mp4")
    print("outputFileURL \(outputFileURL)")
    guard let writer = try? AVAssetWriter(outputURL: outputFileURL, fileType: .mp4) else {
      return nil
    }
    var outputSetting: [String : Any] = [AVVideoCodecKey : AVVideoCodecType.h264]
    if track.preferredTransform.isIdentity ||
        track.preferredTransform.isRotated180 {
      outputSetting[AVVideoHeightKey] = track.naturalSize.height
      outputSetting[AVVideoWidthKey] = track.naturalSize.width
    } else {
      outputSetting[AVVideoHeightKey] = track.naturalSize.width
      outputSetting[AVVideoWidthKey] = track.naturalSize.height
    }
    let input = AVAssetWriterInput(
      mediaType: .video,
      outputSettings: outputSetting)
    guard writer.canAdd(input) else {
      return nil
    }
    writer.add(input)
    writer.startWriting()
    writer.startSession(atSourceTime: sourceTime)
    print("Start writing")
    return (writer, input)
  }

  fileprivate func writeIfNeeded(_ sampleBuffer: CMSampleBuffer) -> Bool {
    var shouldFinishWriting = false
    guard let startIndex = indexFirstDetectedPerson,
          frameIndex >= startIndex else {
      return shouldFinishWriting
    }

    if frameIndex == startIndex,
       #available(iOS 13, *) {
      firstSampleToRecordTime = sampleBuffer.presentationTimeStamp
    }

    let frameRate = firstVideoTrack?.nominalFrameRate ?? 0
    let framesToWrite = Int((numberOfSecondsToRecord * frameRate).rounded(.awayFromZero))
    let framesToWriteOr1 = max(framesToWrite, 1)
    let endIndex = startIndex + framesToWriteOr1 - 1
    guard frameIndex <= endIndex else {
      return shouldFinishWriting
    }

    var endIndexWithoutPerson: Int?
    if let lastIndexWithPerson = indexLastDetectedPerson {
      let framesToWriteWithoutPerson = Int((numberOfSecondsToRecordWithoutPerson * frameRate).rounded(.awayFromZero))
      let endIndex = lastIndexWithPerson + framesToWriteWithoutPerson
      endIndexWithoutPerson = endIndex
      guard frameIndex <= endIndex else {
        shouldFinishWriting = true
        return shouldFinishWriting
      }
    }

    guard let input = writeHelper?.input else {
      return shouldFinishWriting
    }
    while !(input.isReadyForMoreMediaData) {
      sleep(50)
    }
    input.append(sampleBuffer)

    if frameIndex == endIndex {
      shouldFinishWriting = true
    }
    else if let index = endIndexWithoutPerson,
            frameIndex == index {
      shouldFinishWriting = true
    }
    return shouldFinishWriting
  }

  fileprivate func finishWritingIfNeeded() {
    guard let writer = writeHelperBackingStore?.writer,
          writer.status == .writing else {
      return
    }
    writer.finishWriting { [weak self] in
      print("Writing video file done.")
      print("writer.status \(writer.status.rawValue)")
      UISaveVideoAtPathToSavedPhotosAlbum(
        writer.outputURL.path,
        self,
        #selector(self?.video(_:didFinishSavingWith:contextInfo:)),
        nil)
    }
    resetWriteRelated()
  }

  @objc fileprivate func video(
    _ videoPath: String,
    didFinishSavingWith error: Error?,
    contextInfo: UnsafeMutableRawPointer?)
  {
    print("Copying video at \(videoPath) to Photos finished.")
    if let error = error {
      print(error)
    }
    do {
      try FileManager.default.removeItem(atPath: videoPath)
    } catch {
      print(error)
    }
    print("Removed file at \(videoPath).")
  }

  fileprivate func resetWriteRelated() {
    indexFirstDetectedPerson = nil
    indexLastDetectedPerson = nil
    firstSampleToRecordTime = CMTime.zero
    writeHelperBackingStore = nil
  }
}

extension CGAffineTransform {

  fileprivate var isRotated180: Bool {
    return a.effectivelyEquals(-1) &&
      b.effectivelyEquals(0) &&
      c.effectivelyEquals(0) &&
      d.effectivelyEquals(-1)
  }
}

extension CGFloat {

  fileprivate func effectivelyEquals(_ another: CGFloat) -> Bool {
    return (abs(self - another) <= 1e-3)
  }
}

extension ViewController {

  func setUpBoundingBoxViews() {
    for _ in 0..<maxBoundingBoxViews {
      boundingBoxViews.append(BoundingBoxView())
    }

    // The label names are stored inside the MLModel's metadata.
    guard let userDefined = coreMLModel.model.modelDescription.metadata[MLModelMetadataKey.creatorDefinedKey] as? [String: String],
       let allLabels = userDefined["classes"] else {
      fatalError("Missing metadata")
    }

    let labels = allLabels.components(separatedBy: ",")

    // Assign random colors to the classes.
    for label in labels {
      colors[label] = UIColor(red: CGFloat.random(in: 0...1),
                              green: CGFloat.random(in: 0...1),
                              blue: CGFloat.random(in: 0...1),
                              alpha: 1)
    }
  }

  func setUpCamera() {
    videoCapture = VideoCapture()
    videoCapture.delegate = self

    videoCapture.setUp(sessionPreset: .hd1280x720) { success in
      if success {
        // Add the video preview into the UI.
        if let previewLayer = self.videoCapture.previewLayer {
          self.videoPreview.layer.addSublayer(previewLayer)
          self.resizePreviewLayer()
        }

        // Add the bounding box layers to the UI, on top of the video preview.
        for box in self.boundingBoxViews {
          box.addToLayer(self.videoPreview.layer)
        }

        // Once everything is set up, we can start capturing live video.
        self.videoCapture.start()
      }
    }
  }

  override func viewWillLayoutSubviews() {
    super.viewWillLayoutSubviews()
    resizePreviewLayer()
  }

  func resizePreviewLayer() {
    videoCapture.previewLayer?.frame = videoPreview.bounds
  }

  func predict(sampleBuffer: CMSampleBuffer) {
    if currentBuffer == nil, let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) {
      currentBuffer = pixelBuffer

      // Get additional info from the camera.
      var options: [VNImageOption : Any] = [:]
      if let cameraIntrinsicMatrix = CMGetAttachment(sampleBuffer, key: kCMSampleBufferAttachmentKey_CameraIntrinsicMatrix, attachmentModeOut: nil) {
        options[.cameraIntrinsics] = cameraIntrinsicMatrix
      }

      let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: .up, options: options)
      do {
        try handler.perform([self.visionRequest])
      } catch {
        print("Failed to perform Vision request: \(error)")
      }

      currentBuffer = nil
    }
  }

  func processObservations(for request: VNRequest, error: Error?) {
    detectPerson((request.results as? [VNRecognizedObjectObservation]) ?? [])
    DispatchQueue.main.async {
      if let results = request.results as? [VNRecognizedObjectObservation] {
        self.show(predictions: results)
      } else {
        self.show(predictions: [])
      }
    }
  }

  fileprivate func detectPerson(_ observations: [VNRecognizedObjectObservation]) {
    for observation in observations {
      var idToConfidenceMap = [String : VNConfidence]()
      for label in observation.labels {
        idToConfidenceMap[label.identifier] = label.confidence
      }
      guard let confidence = idToConfidenceMap[personID],
            confidence >= minConfidenceToStartRecording else {
        continue
      }
      if indexFirstDetectedPerson == nil {
        indexFirstDetectedPerson = frameIndex
        print("Start recording at frame index \(frameIndex).")
      }
      indexLastDetectedPerson = frameIndex
    }
  }
}

fileprivate let personID = "person"
fileprivate let minConfidenceToStartRecording: VNConfidence = 0.9
fileprivate let numberOfSecondsToRecord: Float = 10
fileprivate let numberOfSecondsToRecordWithoutPerson: Float = 5

extension ViewController {

  func show(predictions: [VNRecognizedObjectObservation]) {
    for i in 0..<boundingBoxViews.count {
      if i < predictions.count {
        let prediction = predictions[i]

        /*
         The predicted bounding box is in normalized image coordinates, with
         the origin in the lower-left corner.

         Scale the bounding box to the coordinate system of the video preview,
         which is as wide as the screen and has a 16:9 aspect ratio. The video
         preview also may be letterboxed at the top and bottom.

         Based on code from https://github.com/Willjay90/AppleFaceDetection

         NOTE: If you use a different .imageCropAndScaleOption, or a different
         video resolution, then you also need to change the math here!
        */

        let width = view.bounds.width
        let height = width * 16 / 9
        let offsetY = (view.bounds.height - height) / 2
        let scale = CGAffineTransform.identity.scaledBy(x: width, y: height)
        let transform = CGAffineTransform(scaleX: 1, y: -1).translatedBy(x: 0, y: -height - offsetY)
        let rect = prediction.boundingBox.applying(scale).applying(transform)

        // The labels array is a list of VNClassificationObservation objects,
        // with the highest scoring class first in the list.
        let bestClass = prediction.labels[0].identifier
        let confidence = prediction.labels[0].confidence

        // Show the bounding box.
        let label = String(format: "%@ %.1f", bestClass, confidence * 100)
        let color = colors[bestClass] ?? UIColor.red
        boundingBoxViews[i].show(frame: rect, label: label, color: color)
      } else {
        boundingBoxViews[i].hide()
      }
    }
  }
}

extension ViewController: VideoCaptureDelegate {
  func videoCapture(_ capture: VideoCapture, didCaptureVideoFrame sampleBuffer: CMSampleBuffer) {
//    predict(sampleBuffer: sampleBuffer)
  }
}
