<a id="top"></a>

<div align=center>
<img src="img\logo.png" width="180px">
</div>
<h2 align="center"><a href="paper/medical_embodied_ai.pdf"> Towards Next-Generation Healthcare: A Survey of Medical Embodied AI for Perception, Decision-Making, and Action </a></h2>
<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv-XXX-orange)](XXX)
![Type-Survey](https://img.shields.io/badge/Type-Survey-blue)
![Topic-Medical%20Embodied%20AI](https://img.shields.io/badge/Topic-Medical%20Embodied%20AI-purple)
[![](https://img.shields.io/badge/Project-%F0%9F%9A%80-pink)](https://github.com/HCPLab-SYSU/Embodied_AI_Paper_List)

</div>

> [!NOTE]
> 🌟 If this project is helpful to you, please click on a Star to support us！

---

## 🏠 About

Foundation models have demonstrated impressive performance in enhancing  healthcare efficiency. However, their limited ability to perceive and interact  with the physical world significantly constrains their utility in real-world clinical  workflows. Recently, embodied artificial intelligence (AI) provides a promising  physical-interactive paradigm for intelligent healthcare by integrating percep tion, decision-making, and action within a closed-loop system. Nevertheless, the  exploration of embodied AI for healthcare is still in its infancy. To support these  advances, this review systematically surveys the key components of embodied AI,  focusing on the integration of perception, decision-making, and action. Addition ally, we present a comprehensive overview of representative medical applications,  relevant datasets, major challenges in clinical practice, and further discuss the  key directions for future research in this emerging field. The associated project  can be found at XXXX.

<div align="center">

### [📖 1. Introduction](#1-introduction) | [🤖 2. Embodied AI](#2-embodied-ai)

### [🏥 3. Embodied AI in Medicine](#3-embodied-ai-in-medicine) | [📊 4. Datasets and benchmark](#4-datasets-and-benchmark)

</div>

## 📖 1. Introduction

<div align=center>
<img src="img\Fig1.jpg">
</div>

**Fig. 1** Foundations of embodied AI. a, Publication volume, temporal trends over the past decade,  and representative keywords related to embodied intelligence. The statistics are obtained from Google  Scholar using “embodied AI” as the search query. b, The four developmental stages of embodied intel ligence, namely the Conceptual Germination Stage, Paradigm Shift Stage, Learning-Driven Stage,  and Large Model Empowered Stage. c, A comparison between disembodied intelligence and embodied  intelligence. Unlike its disembodied counterpart, embodied intelligence is distinguished by its inher ent ability to interact with the environment. d, Core components of embodied intelligence. At the  macroscopic level, it consists of agents and their environments; at the technical level, it encompasses  embodied perception, embodied decision-making, and embodied action.

<div align="center">
<a href="#top">⬆ Back to top</a>
</div>

## 🤖 2. Embodied AI

* [38][**Mind, 1950**] Computing machinery and intelligence [paper](XXX)
* [39][**arXiv, 2025**] Neural brain: A neuroscience-inspired framework for embodied agents [paper](XXX)
* [40][**ACM Computing Surveys, 2025**] Embodied intelligence: A synergy of morphology, action, perception and learning [paper](XXX)
* [41][**ICML, 2024**] Position: a call for embodied AI [paper](XXX)
* [42][**arXiv, 2025**] Embodiedreamer: Advancing real2sim2real transfer for policy training via embodied world modeling [paper](XXX)
* [43][**arXiv, 2024**] Dexsim2real^2: Building explicit world model for precise articulated object dexterous manipulation [paper](XXX)
* [44][**arXiv, 2025**] Bridging the sim2real gap: Vision encoder pre-training for visuomotor policy transfer [paper](XXX)
* [45][**ICML, 2024**] Dexscale: Automating data scaling for sim2real generalizable robot control [paper](XXX)

### 2.1 Embodied Perception

* [90][**CVPR, 2024**] Evidential active recognition: Intelligent and prudent open-world embodied perception [paper](XXX)
* [91][**CVPR, 2024**] Embodiedscan: A holistic multi-modal 3d perception suite towards embodied AI [paper](XXX)
* [92][**Information Fusion, 2024**] Advancements in perception system with multi-sensor fusion for embodied agents [paper](XXX)
* [93][**Information Fusion, 2025**] Tactile data generation and applications based on visuo-tactile sensors: A review [paper](XXX)

#### 2.1.1 Object Perception

* [94][**IEEE Transactions on Cognitive and Developmental Systems, 2020**] Robot multimodal object perception and recognition: Synthetic maturation of sensorimotor learning in embodied systems [paper](XXX)
* [95][**IEEE Transactions on Robotics, 2025**] Predictive visuo-tactile interactive perception framework for object properties inference [paper](XXX)
* [96][**Proceedings of the IEEE, 2002**] Gradient-based learning applied to document recognition [paper](XXX)
* [97][**NeurIPS, 2017**] Attention is all you need [paper](XXX)
* [98][**ICCV, 2023**] Segment anything [paper](XXX)
* [99][**arXiv, 2024**] Sam 2: Segment anything in images and videos [paper](XXX)
* [100][**ICCV, 2021**] Emerging properties in self-supervised vision transformers [paper](XXX)
* [101][**arXiv, 2023**] Dinov2: Learning robust visual features without supervision [paper](XXX)
* [102][**arXiv, 2025**] Dinov3 [paper](XXX)
* [46][**NeurIPS, 2012**] Imagenet classification with deep convolutional neural networks [paper](XXX)
* [47][**arXiv, 2014**] Very deep convolutional networks for large-scale image recognition [paper](XXX)
* [48][**CVPR, 2016**] Deep residual learning for image recognition [paper](XXX)
* [49][**NeurIPS, 2015**] Faster R-CNN: Towards real-time object detection with region proposal networks [paper](XXX)
* [50][**CVPR, 2016**] You only look once: Unified, real-time object detection [paper](XXX)
* [51][**MICCAI, 2015**] U-net: Convolutional networks for biomedical image segmentation [paper](XXX)
* [52][**Seminal Graphics Papers, 2023**] SMPL: A skinned multi-person linear model [paper](XXX)
* [53][**ACCV, 2014**] 3d human pose estimation from monocular images with deep convolutional neural network [paper](XXX)
* [54][**CVPR, 2021**] Open-vocabulary object detection using captions [paper](XXX)
* [55][**ECCV, 2022**] Open vocabulary object detection with pseudo bounding-box labels [paper](XXX)

#### 2.1.2 Scene Perception

* [103][**CVPR, 2025**] Embodied scene understanding for vision language models via metavqa [paper](XXX)
* [104][**ECCV, 2024**] Embodied understanding of driving scenarios [paper](XXX)
* [105][**WACV, 2025**] Scene-llm: Extending language model for 3d visual reasoning [paper](XXX)
* [106][**ECCV, 2024**] Sceneverse: Scaling 3d vision-language learning for grounded scene understanding [paper](XXX)
* [107][**IROS, 2024**] Mm3dgs slam: Multi-modal 3d gaussian splatting for slam using vision, depth, and inertial measurements [paper](XXX)
* [108][**IEEE TPAMI, 2022**] Learning view-based graph convolutional network for multi-view 3d shape analysis [paper](XXX)
* [56][**ISPRS Journal of Photogrammetry and Remote Sensing, 2024**] Few-shot remote sensing image scene classification: Recent advances, new baselines, and future trends [paper](XXX)
* [57][**WACV, 2024**] U3ds3: Unsupervised 3d semantic scene segmentation [paper](XXX)
* [58][**IEEE TPAMI, 2024**] Etpnav: Evolving topological planning for vision-language navigation in continuous environments [paper](XXX)
* [59][**ICRA, 2024**] Robohop: Segment-based topological map representation for open-world visual navigation [paper](XXX)
* [60][**arXiv, 2025**] Panorama: The rise of omnidirectional vision in the embodied AI era [paper](XXX)
* [61][**IROS, 2024**] Omninxt: A fully open-source and compact aerial robot with omnidirectional visual perception [paper](XXX)

#### 2.1.3 Behavior Perception

* [109][**Expert Systems with Applications, 2024**] Human activity recognition with smartphone-integrated sensors: A survey [paper](XXX)
* [110][**Artificial Intelligence Review, 2024**] A survey of video-based human action recognition in team sports [paper](XXX)
* [111][**Expert Systems with Applications, 2024**] A new framework for deep learning video based human action recognition on the edge [paper](XXX)
* [112][**CVPR, 2024**] Blockgcn: Redefine topology awareness for skeleton-based action recognition [paper](XXX)
* [113][**IEEE Transactions on Image Processing, 2024**] Learnable feature augmentation framework for temporal action localization [paper](XXX)
* [114][**Pattern Recognition, 2025**] Sam-net: Semantic-assisted multimodal network for action recognition in rgb-d videos [paper](XXX)
* [115][**IEEE Transactions on Information Forensics and Security, 2025**] Collaboratively self-supervised video representation learning for action recognition [paper](XXX)
* [62][**ICRA, 2024**] Anticipate & act: Integrating LLMs and classical planning for efficient task execution in household environments [paper](XXX)
* [63][**ICRA, 2025**] Castl: Constraints as specifications through LLM translation for long-horizon task and motion planning [paper](XXX)

#### 2.1.4 Expression Perception

* [116][**Proceedings of the IEEE, 2023**] Facial micro-expressions: An overview [paper](XXX)
* [117][**IEEE Transactions on Instrumentation and Measurement, 2023**] Understanding deep learning techniques for recognition of human emotions using facial expressions: A comprehensive survey [paper](XXX)
* [118][**arXiv, 2025**] Multimodal emotion recognition in conversations: A survey of methods, trends, challenges and prospects [paper](XXX)
* [119][**IEEE Transactions on Affective Computing, 2025**] Mer-clip: Au-guided vision-language alignment for micro-expression recognition [paper](XXX)
* [64][**IEEE TPAMI, 2024**] Prompt tuning of deep neural networks for speaker-adaptive visual speech recognition [paper](XXX)
* [65][**Pattern Recognition, 2025**] Context transformer with multiscale fusion for robust facial emotion recognition [paper](XXX)
* [66][**IEEE Transactions on Consumer Electronics, 2025**] Meta-transfer learning based cross-domain gesture recognition using wifi channel state information [paper](XXX)
* [67][**Scientific Data, 2025**] EMG dataset for gesture recognition with arm translation [paper](XXX)
* [68][**CVPR, 2025**] Uncertain multimodal intention and emotion understanding in the wild [paper](XXX)

### 2.2 Embodied Decision-Making

* [120][**arXiv, 2025**] A comprehensive survey on multi-agent cooperative decision-making: Scenarios, approaches, challenges and perspectives [paper](XXX)

#### 2.2.1 Task Planning

* [121][**ACM Computing Surveys, 2023**] Recent trends in task and motion planning for robotics: A survey [paper](XXX)
* [122][**IEEE/ASME Transactions on Mechatronics, 2024**] A survey of optimization-based task and motion planning: From classical to learning approaches [paper](XXX)
* [123][**ICRA, 2025**] Guiding long-horizon task and motion planning with vision language models [paper](XXX)
* [69][**IROS, 2021**] Learning symbolic operators for task and motion planning [paper](XXX)
* [70][**Journal of Artificial Intelligence Research, 2003**] PDDL2.1: An extension to PDDL for expressing temporal planning domains [paper](XXX)
* [71][**ICRA, 2025**] Delta: Decomposed efficient long-term robot task planning using large language models [paper](XXX)
* [72][**ICRA, 2025**] Fast and accurate task planning using neuro-symbolic language models and multi-level goal decomposition [paper](XXX)

#### 2.2.2 Embodied Navigation

* [124][**Science China Information Sciences, 2025**] Embodied navigation [paper](XXX)
* [125][**Information Fusion, 2024**] Embodied navigation with multi-modal information: A survey from tasks to methodology [paper](XXX)
* [126][**CVPR, 2025**] Mne-slam: Multi-agent neural slam for mobile robots [paper](XXX)
* [127][**Information Sciences, 2025**] Mahaco: Multi-algorithm hybrid ant colony optimizer for 3d path planning of a group of uavs [paper](XXX)
* [128][**IEEE Transactions on Automation Science and Engineering, 2024**] A survey of object goal navigation [paper](XXX)
* [129][**ICRA, 2024**] Collision avoidance and navigation for a quadrotor swarm using end-to-end deep reinforcement learning [paper](XXX)
* [130][**ICRA, 2024**] Uivnav: Underwater information-driven vision-based navigation via imitation learning [paper](XXX)
* [131][**Information Fusion, 2024**] Macns: A generic graph neural network integrated deep reinforcement learning based multi-agent collaborative navigation system for dynamic trajectory planning [paper](XXX)
* [132][**AAAI, 2025**] Naviformer: A spatio-temporal context-aware transformer for object navigation [paper](XXX)
* [133][**CVPR, 2025**] Towards long-horizon vision-language navigation: Platform, benchmark and method [paper](XXX)
* [73][**IEEE TPAMI, 2025**] Gaussnav: Gaussian splatting for visual navigation [paper](XXX)
* [74][**IEEE TPAMI, 2025**] Constraint-aware zero-shot vision-language navigation in continuous environments [paper](XXX)

#### 2.2.3 Embodied Question Answering (EQA)

* [134][**arXiv, 2025**] Embodied intelligence for 3d understanding: A survey on 3d scene question answering [paper](XXX)
* [75][**arXiv, 2024**] GraphEQA: Using 3d semantic scene graphs for real-time embodied question answering [paper](XXX)
* [76][**CVPR, 2024**] OpenEQA: Embodied question answering in the era of foundation models [paper](XXX)

### 2.3 Embodied Action

#### 2.3.1 Imitation Learning-Based Action

* [135][**IEEE Transactions on Cybernetics, 2024**] A survey of imitation learning: Algorithms, recent developments, and challenges [paper](XXX)
* [136][**Foundations and Trends in Robotics, 2018**] An algorithmic perspective on imitation learning [paper](XXX)
* [137][**IEEE Transactions on Industrial Electronics, 2025**] Deep multimodal imitation learning-based framework for robot-assisted medical examination [paper](XXX)
* [138][**ICRA, 2025**] Egomimic: Scaling imitation learning via egocentric video [paper](XXX)
* [77][**NeurIPS, 2024**] Is behavior cloning all you need? understanding horizon in imitation learning [paper](XXX)
* [78][**IEEE Robotics and Automation Letters, 2025**] Stable-bc: Controlling covariate shift with stable behavior cloning [paper](XXX)
* [79][**AAAI, 2025**] Inverse reinforcement learning by estimating expertise of demonstrators [paper](XXX)
* [80][**ICLR, 2025**] Understanding constraint inference in safety-critical inverse reinforcement learning [paper](XXX)

#### 2.3.2 Reinforcement Learning-Based Action

* [139][**AAAI, 2025**] Deep reinforcement learning for robotics: A survey of real-world successes [paper](XXX)
* [140][**IEEE Transactions on Neural Networks and Learning Systems, 2022**] Deep reinforcement learning: A survey [paper](XXX)
* [141][**Journal of Artificial Intelligence Research, 1996**] Reinforcement learning: A survey [paper](XXX)
* [142][**AAAI, 2025**] Autonomous option invention for continual hierarchical reinforcement learning and planning [paper](XXX)
* [143][**IEEE Transactions on Intelligent Transportation Systems, 2025**] Toward adaptive and coordinated transportation systems: A multi-personality multi-agent meta-reinforcement learning framework [paper](XXX)
* [144][**Nature Machine Intelligence, 2025**] Model-based reinforcement learning for ultrasound-driven autonomous microrobots [paper](XXX)
* [81][**Artificial Intelligence for Engineers, 2025**] Value-based reinforcement learning [paper](XXX)
* [82][**Machine Learning, 1992**] Q-learning [paper](XXX)
* [83][**arXiv, 2017**] Proximal policy optimization algorithms [paper](XXX)
* [84][**ICML, 2016**] Asynchronous methods for deep reinforcement learning [paper](XXX)
* [85][**arXiv, 2015**] Continuous control with deep reinforcement learning [paper](XXX)
* [86][**ICML, 2018**] Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor [paper](XXX)

#### 2.3.3 Large Model-Driven Action

* [87][**arXiv, 2023**] GPT-4 technical report [paper](XXX)
* [88][**NeurIPS, 2022**] Flamingo: a visual language model for few-shot learning [paper](XXX)
* [89][**CoRL, 2023**] Rt-2: Vision-language-action models transfer web knowledge to robotic control [paper](XXX)
* [145][**arXiv, 2023**] Palm-e: An embodied multimodal language model [paper](XXX)
* [146][**ICML, 2023**] Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models [paper](XXX)
* [147][**arXiv, 2022**] Code as policies: Language model programs for embodied control [paper](XXX)
* [148][**arXiv, 2025**] Embodied AI agents: Modeling the world [paper](XXX)

<div align="center">
<a href="#top">⬆ Back to top</a>
</div>

## 🏥 3. Embodied AI in Medicine

<div align=center>
<img src="img\Fig2.png">
</div>

**Fig. 2** Embodied AI in medicine. Corresponding to the core components of embodied AI, med ical embodied AI encompasses medical embodied perception, medical embodied decision-making,  and medical embodied action. a, Medical embodied perception includes medical instrument and  organ recognition, perception and modeling of surgical and clinical environments, detection of medi cal operational behaviors, and understanding of affective and interactive cues. b, Medical embodied  decision-making encompasses medical workflow modeling and task planning, medical navigation sys tems, and clinical question-answering and decision-support mechanisms. c, Medical embodied action  consists of imitation-based medical actions, reinforcement-based medical actions, and large-model driven medical actions.

### 3.1 Medical Embodied Perception

#### 3.1.1 Medical Instrument and Organ Recognition

* [149][**arXiv, 2020**] Deep learning in multi-organ segmentation [paper](XXX)
* [150][**Artificial Intelligence Review, 2024**] Deep learning for surgical instrument recognition and segmentation in robotic-assisted surgeries: a systematic review [paper](XXX)
* [151][**IEEE Transactions on Neural Networks and Learning Systems, 2022**] SwinPA-Net: Swin transformer-based multiscale feature pyramid aggregation network for medical image segmentation [paper](XXX)
* [152][**Medical Image Analysis, 2021**] ST-MTL: Spatio-temporal multitask learning model to predict scanpath while tracking instruments in robotic surgery [paper](XXX)
* [153][**IEEE Robotics and Automation Letters, 2019**] Deep learning based robotic tool detection and articulation estimation with spatio-temporal layers [paper](XXX)
* [154][**arXiv, 2025**] SurgVLM: A large vision-language model and systematic evaluation benchmark for surgical intelligence [paper](XXX)

#### 3.1.2 Surgical and Clinical Environment Perception and Modeling

* [155][**arXiv, 2020**] A robotic 3D perception system for operating room environment awareness [paper](XXX)
* [156][**Neurosurgical Focus, 2024**] Creation of a microsurgical neuroanatomy laboratory and virtual operating room: a preliminary study [paper](XXX)
* [157][**IJCARS, 2025**] NeRF-OR: neural radiance fields for operating room scene reconstruction from sparse-view RGB-D videos [paper](XXX)
* [158][**MICCAI, 2024**] Deform3DGS: Flexible deformation for fast surgical scene reconstruction with gaussian splatting [paper](XXX)
* [159][**MICCAI, 2022**] 4D-OR: Semantic scene graphs for or domain modeling [paper](XXX)
* [160][**MICCAI, 2023**] Labrad-OR: Lightweight memory scene graphs for accurate bimodal reasoning in dynamic operating rooms [paper](XXX)
* [161][**arXiv, 2025**] Spatial-ORMLLM: Improve spatial relation understanding in the operating room with multimodal large language model [paper](XXX)

#### 3.1.3 Medical Operation Behavior Detection

* [162][**IEEE JBHI, 2023**] Deep learning in surgical workflow analysis: a review of phase and step recognition [paper](XXX)
* [163][**IEEE FG, 2024**] MGRFormer: A multimodal transformer approach for surgical gesture recognition [paper](XXX)
* [164][**ITNEC, 2024**] Surgical gesture recognition in open surgery based on 3DCNN and SlowFast [paper](XXX)
* [165][**CGI, 2024**] TransSG: A spatial-temporal transformer for surgical gesture recognition [paper](XXX)
* [166][**Annals of the New York Academy of Sciences, 2025**] STANet: A surgical gesture recognition method based on spatiotemporal fusion [paper](XXX)
* [167][**IEEE Access, 2024**] Audio-and video-based human activity recognition systems in healthcare [paper](XXX)
* [168][**IEEE Transactions on Medical Imaging, 2022**] Gesture recognition in robotic surgery with multimodal attention [paper](XXX)

#### 3.1.4 Emotional Interaction Understanding

* [169][**IEEE Transactions on Affective Computing, 2020**] Deep facial expression recognition: A survey [paper](XXX)
* [170][**IEEE Transactions on Affective Computing, 2022**] Deep learning for micro-expression recognition: A survey [paper](XXX)
* [171][**Pattern Recognition, 2025**] Multimodal latent emotion recognition from micro-expression and physiological signal [paper](XXX)
* [172][**Information Fusion, 2025**] Towards facial micro-expression detection and classification using modified multimodal ensemble learning approach [paper](XXX)
* [173][**IEEE Transactions on Circuits and Systems for Video Technology, 2024**] Dep-former: Multimodal depression recognition based on facial expressions and audio features via emotional changes [paper](XXX)
* [174][**Expert Systems with Applications, 2024**] MSER: Multimodal speech emotion recognition using cross-attention with deep fusion [paper](XXX)
* [175][**IEEE JBHI, 2025**] Multimodal fusion of behavioral and physiological signals for enhanced emotion recognition via feature decoupling and knowledge transfer [paper](XXX)
* [176][**IEEE Transactions on Instrumentation and Measurement, 2024**] Deep learning-based automated emotion recognition using multimodal physiological signals and time-frequency methods [paper](XXX)
* [177][**MICCAI, 2025**] MedVLM-RL: Incentivizing medical reasoning capability of vision-language models (VLMs) via reinforcement learning [paper](XXX)
* [178][**arXiv, 2023**] DialogueLLM: Context and emotion knowledge-tuned large language models for emotion recognition in conversations [paper](XXX)

### 3.2 Medical Embodied Decision-Making

* [179][**EBioMedicine, 2019**] Artificial intelligence to support clinical decision-making processes [paper](XXX)
* [180][**JAMA Surgery, 2020**] Artificial intelligence and surgical decision-making [paper](XXX)

#### 3.2.1 Medical Workflow Modeling and Task Planning

* [181][**BMC Oral Health, 2025**] Artificial intelligence and augmented reality for guided implant surgery planning: a proof of concept [paper](XXX)
* [182][**Health Systems, 2021**] Clinical pathway modelling: a literature review [paper](XXX)
* [183][**MICCAI, 2021**] Trans-SVNet: Accurate phase recognition from surgical videos via hybrid embedding aggregation transformer [paper](XXX)
* [184][**MICCAI, 2020**] TeCNO: Surgical phase recognition with multi-stage temporal convolutional networks [paper](XXX)
* [185][**IJCARS, 2022**] PATG: position-aware temporal graph networks for surgical phase recognition on laparoscopic videos [paper](XXX)
* [186][**IEEE-EMBS BHI, 2022**] Towards graph representation learning based surgical workflow anticipation [paper](XXX)
* [187][**NeurIPS, 2023**] LLaVA-Med: Training a large language-and-vision assistant for biomedicine in one day [paper](XXX)

#### 3.2.2 Medical Navigation Systems

* [188][**Langenbeck's Archives of Surgery, 2013**] Navigation in surgery [paper](XXX)
* [189][**International Journal of Nanomedicine, 2025**] Localized drug delivery in different gastrointestinal cancers: navigating challenges and advancing nanotechnological solutions [paper](XXX)
* [190][**ICSR, 2024**] Utilizing a social robot as a greeter at a children's hospital [paper](XXX)
* [191][**IPIN, 2015**] Navigating in large hospitals [paper](XXX)
* [192][**Neurosurgery, 1999**] BrainLAB VectorVision neuronavigation system: technology and clinical experiences in 131 cases [paper](XXX)
* [193][**Annals of Biomedical Engineering, 2021**] A wearable augmented reality navigation system for surgical telementoring based on Microsoft HoloLens [paper](XXX)
* [194][**IEEE Transactions on Automation Science and Engineering, 2025**] RL-USRegi: Autonomous ultrasound registration for radiation-free spinal surgical navigation using reinforcement learning [paper](XXX)
* [195][**IJCARS, 2024**] Autonomous navigation of catheters and guidewires in mechanical thrombectomy using inverse reinforcement learning [paper](XXX)
* [196][**AAAI, 2024**] NavGPT: Explicit reasoning in vision-and-language navigation with large language models [paper](XXX)
* [197][**ECCV, 2024**] NavGPT-2: Unleashing navigational reasoning capability for large vision-language models [paper](XXX)

#### 3.2.3 Clinical Question-answering and Decision Support

* [198][**Nature Medicine, 2025**] Toward expert-level medical question answering with large language models [paper](XXX)
* [199][**Artificial Intelligence in Medicine, 2023**] Medical visual question answering: A survey [paper](XXX)
* [200][**ACM Computing Surveys, 2022**] Biomedical question answering: a survey of approaches and challenges [paper](XXX)
* [201][**Nature Medicine, 2025**] Clinical implementation of an AI-based prediction model for decision support for patients undergoing colorectal cancer surgery [paper](XXX)
* [202][**BMC Oral Health, 2025**] Artificial intelligence-based chatbot assistance in clinical decision-making for medically complex patients in oral surgery [paper](XXX)
* [203][**Journal of Biology and Health Science, 2025**] Multimodal decision support system for improved diagnosis and healthcare decision making [paper](XXX)
* [204][**Computerized Medical Imaging and Graphics, 2025**] MedBlip: A multimodal method of medical question-answering based on fine-tuning large language model [paper](XXX)

### 3.3 Medical Embodied Action

* [205][**Science Robotics, 2025**] Will your next surgeon be a robot? Autonomy and AI in robotic surgery [paper](XXX)
* [206][**Annual Review of Control, Robotics, and Autonomous Systems, 2021**] Autonomy in surgical robotics [paper](XXX)

#### 3.3.1 Medical Imitation-based Action

* [207][**IEEE Transactions on Biomedical Engineering, 2025**] Imitation learning for path planning in cardiac percutaneous interventions [paper](XXX)
* [208][**arXiv, 2024**] Surgical robot transformer (SRT): Imitation learning for surgical tasks [paper](XXX)
* [209][**ICRA, 2021**] Intermittent visual servoing: Efficiently learning policies robust to instrument changes for high-precision surgical manipulation [paper](XXX)
* [210][**arXiv, 2025**] SuFIA-BC: Generating high quality demonstration data for visuomotor policy learning in surgical subtasks [paper](XXX)
* [211][**IEEE Transactions on Biomedical Engineering, 2021**] Inverse reinforcement learning intra-operative path planning for steerable needle [paper](XXX)
* [212][**ICRA, 2020**] Collaborative robot-assisted endovascular catheterization with generative adversarial imitation learning [paper](XXX)
* [213][**IROS, 2024**] Towards a surgeon-in-the-loop ophthalmic robotic apprentice using reinforcement and imitation learning [paper](XXX)
* [214][**ICRA, 2022**] 3d perception based imitation learning under limited demonstration for laparoscope control in robotic surgery [paper](XXX)

#### 3.3.2 Medical Reinforcement-based Action

* [215][**ACM Computing Surveys, 2021**] Reinforcement learning in healthcare: A survey [paper](XXX)
* [216][**RO-MAN, 2020**] Collaborative suturing: A reinforcement learning approach to automate hand-off task in suturing for surgical robots [paper](XXX)
* [217][**Journal of Machine Learning Research, 2023**] LapGym-an open source framework for reinforcement learning in robot-assisted laparoscopic surgery [paper](XXX)
* [218][**ICMA, 2022**] Evaluation of an autonomous navigation method for vascular interventional surgery in virtual environment [paper](XXX)
* [219][**arXiv, 2024**] Surgical task automation using actor-critic frameworks and self-supervised imitation learning [paper](XXX)
* [220][**IEEE Transactions on Industrial Electronics, 2023**] CASOG: Conservative actor–critic with smooth gradient for skill learning in robot-assisted intervention [paper](XXX)

#### 3.3.3 Medical Large Model-Driven Action

* [221][**Nature Reviews Electrical Engineering, 2025**] Innovating robot-assisted surgery through large vision models [paper](XXX)
* [222][**arXiv, 2024**] GP-VLS: A general-purpose vision language model for surgery [paper](XXX)
* [223][**Artificial Intelligence Review, 2024**] Large language models in healthcare: from a systematic review on medical examinations to a comparative analysis on fundamentals of robotic surgery online test [paper](XXX)
* [224][**MICCAI, 2023**] SurgicalGPT: end-to-end language-vision GPT for visual question answering in surgery [paper](XXX)
* [225][**ML4H, 2023**] Med-Flamingo: a multimodal medical few-shot learner [paper](XXX)
* [226][**arXiv, 2024**] RoboNurse-VLA: Robotic scrub nurse system based on vision-language-action model [paper](XXX)

<p align="right"><a href="#top">⬆ Back to top</a></p>

### 3.4 Integrated Application Scenarios in Healthcare

#### 3.4.1 Surgical Robot

* [227][**IEEE Robotics & Automation Magazine, 2021**] Accelerating surgical robotics research: A review of 10 years with the da Vinci research kit [paper](XXX)
* [228][**IJMRCAS, 2013**] Technical review of the da Vinci surgical telemanipulator [paper](XXX)
* [229][**Intuit. Surg., 2013**] da Vinci surgical system [paper](XXX)
* [230][**IJMRCAS, 2015**] A pneumatic laparoscope holder controlled by head movement [paper](XXX)
* [231][**Acta Neurochirurgica, 2016**] Minimally invasive transforaminal lumbar interbody fusion with the ROSA™ spine robot and intraoperative flat-panel CT guidance [paper](XXX)
* [232][**Nature Biomedical Engineering, 2018**] First-in-human study of the safety and viability of intraocular robotic surgery [paper](XXX)
* [233][**Clinical Orthopaedics and Related Research, 1998**] Primary and revision total hip replacement using the ROBODOC (R) system [paper](XXX)
* [234][**Surgical Innovation, 2024**] The use of the Symani Surgical System® in emergency hand trauma care [paper](XXX)
* [235][**Technology in Cancer Research & Treatment, 2010**] The CyberKnife® robotic radiosurgery system in 2010 [paper](XXX)
* [236][**ICRA, 2019**] Robotic bronchoscopy drive mode of the Auris Monarch platform [paper](XXX)
* [237][**European Archives of Oto-Rhino-Laryngology, 2015**] Transoral robotic surgery (TORS) with the Medrobotics Flex™ system: first surgical application on humans [paper](XXX)
* [238][**Journal of Neurosurgery, 2019**] Neuroendovascular-specific engineering modifications to the CorPath GRX robotic system [paper](XXX)

#### 3.4.2 Intelligent Caregiving and Companion Robot

* [239][**BMC Geriatrics, 2019**] The benefits of and barriers to using a social robot PARO in care settings: a scoping review [paper](XXX)
* [240][**European Journal of Pediatrics, 2022**] The pilot study of group robot intervention on pediatric inpatients and their caregivers, using 'new aibo' [paper](XXX)
* [241][**Envisioning the Future of Health Informatics and Digital Health, 2025**] The Pepper robot in healthcare: A scoping review [paper](XXX)
* [242][**Journal of Aging Research & Lifestyle, 2024**] ElliQ, an AI-driven social robot to alleviate loneliness: progress and lessons learned [paper](XXX)
* [243][**Journal of Engineering in Medicine, 2018**] Arash: A social robot buddy to support children with cancer in a hospital environment [paper](XXX)
* [244][**IJERPH, 2020**] Robotics utilization for healthcare digitization in global COVID-19 management [paper](XXX)
* [245][**RO-MAN, 2012**] Technical improvements of the Giraff telepresence robot based on users' evaluation [paper](XXX)
* [246][**SIGGRAPH Emerging Technologies, 2011**] Telenoid: Tele-presence android for communication [paper](XXX)

#### 3.4.3 Immersive Medical Education Platform

* [247][**Handbook of Research on Educational Communications and Technology, 2008**] Cognitive task analysis [paper](XXX)
* [248][**PhD Thesis/Report, 2020**] The first experience of using the Body Interact simulation interactive training platform as a part of interns' attestation [paper](XXX)
* [249][**Surgical Technology International, 2019**] Validation of the hip arthroscopy module of the VirtaMed virtual reality arthroscopy trainer [paper](XXX)
* [250][**Digital Diagnostics, 2024**] Possibilities for using the Vimedix 3.2 virtual simulator to train ultrasound specialists [paper](XXX)
* [251][**CVPR, 2022**] OSSO: Obtaining skeletal shape from outside [paper](XXX)
* [252][**JMLA, 2022**] 3D Organon VR Anatomy [paper](XXX)

#### 3.4.4 Telecollaborative Diagnostic and Treatment System

* [253][**Web, 2025**] Teladoc Health [paper](XXX)
* [254][**Nurse Leader, 2016**] Mercy virtual nursing: An innovative care delivery model [paper](XXX)
* [255][**Molecular Nutrition & Food Research, 2019**] Duck egg white–derived peptide VSEE (Val-Ser-Glu-Glu) regulates bone and lipid metabolisms by Wnt/β-catenin signaling pathway and intestinal microbiota [paper](XXX)
* [256][**Frontiers in Human Neuroscience, 2024**] A retrospective, observational study of real-world clinical data from the cognitive function development therapy program [paper](XXX)

<div align="center">
<a href="#top">⬆ Back to top</a>
</div>

## 📊 4. Datasets and benchmark

### 4.1 Perception Datasets

#### 4.1.1 Organ–Instrument Recognition Datasets

* [257][**Scientific Data, 2022**] ISLES 2022: A multi-center magnetic resonance imaging stroke lesion segmentation dataset [paper](XXX)
* [258][**arXiv, 2023**] The state-of-the-art 3d anisotropic intracranial hemorrhage segmentation on non-contrast head ct: The instance challenge [paper](XXX)
* [259][**arXiv, 2023**] FairSeg: A large-scale medical image segmentation dataset for fairness learning using segment anything model with fair error-bound scaling [paper](XXX)
* [260][**Information Sciences, 2019**] Diagnostic assessment of deep learning algorithms for diabetic retinopathy screening [paper](XXX)
* [261][**Medical Image Analysis, 2022**] AtrialJSQnet: a new framework for joint segmentation and quantification of left atrium and scars incorporating spatial and shape information [paper](XXX)
* [262][**ATM'22 Challenge, 2023**] Multi-site, multi-domain airway tree modeling (ATM'22): A public benchmark for pulmonary airway segmentation [paper](XXX)
* [263][**IPTA, 2020**] SegTHOR: Segmentation of thoracic organs at risk in CT images [paper](XXX)
* [264][**arXiv, 2025**] Tumor detection, segmentation and classification challenge on automated 3d breast ultrasound: The TDSC-ABUS challenge [paper](XXX)
* [265][**Web, 2010**] 3D image reconstruction for comparison of algorithm database (3D-IRCADb-01) [paper](XXX)
* [266][**Medical Image Analysis, 2021**] VerSe: a vertebrae labelling and segmentation benchmark for multi-detector CT images [paper](XXX)
* [267][**Scientific Data, 2024**] Lumbar spine segmentation in MR images: a dataset and a public benchmark [paper](XXX)
* [268][**arXiv, 2021**] CTSpine1K: A large-scale dataset for spinal vertebrae segmentation in computed tomography [paper](XXX)
* [269][**IJCARS, 2021**] Deep learning to segment pelvic bones: large-scale CT datasets and baseline models [paper](XXX)
* [270][**arXiv, 2016**] Skin lesion analysis toward melanoma detection: A challenge at the international symposium on biomedical imaging (ISBI) 2016 [paper](XXX)
* [271][**Expert Systems with Applications, 2015**] MED-NODE: A computer-assisted melanoma diagnosis system using non-dermoscopic images [paper](XXX)
* [272][**Dermoscopy Image Analysis, 2015**] PH2: A public database for the analysis of dermoscopic images [paper](XXX)
* [273][**Data in Brief, 2020**] PAD-UFES-20: A skin lesion dataset composed of patient data and clinical images collected from smartphones [paper](XXX)
* [274][**bioRxiv, 2022**] A web-scraped skin image database of monkeypox, chickenpox, smallpox, cowpox, and measles [paper](XXX)
* [275][**Radiology: Artificial Intelligence, 2023**] TotalSegmentator: robust segmentation of 104 anatomic structures in CT images [paper](XXX)
* [276][**Medical Image Analysis, 2025**] The ULS23 challenge: A baseline model and benchmark dataset for 3D universal lesion segmentation in computed tomography [paper](XXX)
* [277][**Scientific Data, 2022**] A whole-body FDG-PET/CT dataset with manually annotated tumor lesions [paper](XXX)
* [278][**arXiv, 2019**] 2017 robotic instrument segmentation challenge [paper](XXX)
* [279][**IROS, 2020**] LC-GAN: Image-to-image translation based on generative adversarial network for endoscopic images [paper](XXX)
* [280][**arXiv, 2024**] SegStrong-C: Segmenting surgical tools robustly on non-adversarial generated corruptions–an Endovis' 24 challenge [paper](XXX)
* [281][**IEEE Transactions on Medical Imaging, 2016**] EndoNet: a deep architecture for recognition tasks on laparoscopic videos [paper](XXX)
* [282][**CVPR, 2025**] CholecTrack20: A multi-perspective tracking dataset for surgical tools [paper](XXX)
* [283][**arXiv, 2023**] The EndoScapes dataset for surgical scene segmentation, object detection, and critical view of safety assessment [paper](XXX)
* [284][**arXiv, 2020**] m2caiSeg: Semantic segmentation of laparoscopic images using convolutional neural networks [paper](XXX)
* [285][**arXiv, 2021**] FetReg: Placental vessel segmentation and registration in fetoscopy challenge dataset [paper](XXX)

#### 4.1.2 Medical Scene Modeling Datasets

* [286][**CVPR, 2025**] MM-OR: A large multimodal operating room dataset for semantic understanding of high-intensity surgical environments [paper](XXX)
* [287][**IEEE Transactions on Biomedical Engineering, 2016**] See it with your own eyes: Markerless mobile augmented reality for radiation awareness in the hybrid room [paper](XXX)
* [288][**Engineering Applications of Artificial Intelligence, 2023**] Object detection in hospital facilities: A comprehensive dataset and performance evaluation [paper](XXX)
* [289][**Data in Brief, 2018**] MCIndoor20000: A fully-labeled image dataset to advance indoor objects detection [paper](XXX)
* [290][**Data in Brief, 2020**] MyNursingHome: A fully-labelled image dataset for indoor object classification [paper](XXX)

#### 4.1.3 Clinical Action and Pose Estimation Datasets

* [291][**arXiv, 2018**] MVOR: A multi-view RGB-D operating room dataset for 2d and 3d human pose estimation [paper](XXX)
* [292][**IEEE Journal of Translational Engineering in Health and Medicine, 2018**] Patient-specific pose estimation in clinical environments [paper](XXX)
* [293][**Algorithms, 2024**] MMD-MSD: A multimodal multisensory dataset in support of research and technology development for musculoskeletal disorders [paper](XXX)
* [294][**MICCAI Workshops, 2025**] SurgTrack: CAD-free 3D tracking of real-world surgical instruments [paper](XXX)

#### 4.1.4 Multimodal Affective Perception Datasets

* [295][**arXiv, 2025**] Advancing face-to-face emotion communication: A multimodal dataset (AffEc) [paper](XXX)
* [296][**NeurIPS, 2024**] Emotion-LLaMA: Multimodal emotion recognition and reasoning with instruction tuning [paper](XXX)
* [297][**Scientific Data, 2024**] A multimodal dataset for mixed emotion recognition [paper](XXX)
* [298][**IEEE Transactions on Affective Computing, 2024**] SEED-VII: A multimodal dataset of six basic emotions with continuous labels for emotion recognition [paper](XXX)
* [299][**IEEE Transactions on Affective Computing, 2016**] ASCERTAIN: Emotion and personality recognition using commercial sensors [paper](XXX)
* [300][**IEEE JBHI, 2017**] DREAMER: A database for emotion recognition through EEG and ECG signals from wireless low-cost off-the-shelf devices [paper](XXX)

### 4.2 Decision-Making Datasets

#### 4.2.1 Surgical Workflow Annotation Datasets

* [301][**ECCV, 2024**] OphNet: A large-scale video benchmark for ophthalmic surgical workflow understanding [paper](XXX)
* [302][**MICCAI, 2022**] AutoLaparo: A new dataset of integrated multi-tasks for image-guided surgical automation in laparoscopic hysterectomy [paper](XXX)
* [303][**Scientific Data, 2025**] LapEx: A new multimodal dataset for context recognition and practice assessment in laparoscopic surgery [paper](XXX)
* [304][**Computer Methods and Programs in Biomedicine, 2021**] Micro-surgical anastomose workflow recognition challenge report [paper](XXX)

#### 4.2.2 Medical Navigation Datasets

* [305][**IEEE Sensors Journal, 2025**] A portable 6D surgical instrument magnetic localization system with dynamic error correction [paper](XXX)
* [306][**Scientific Data, 2024**] Head model dataset for mixed reality navigation in neurosurgical interventions for intracranial lesions [paper](XXX)
* [307][**CVPR, 2018**] Gibson Env: Real-world perception for embodied agents [paper](XXX)
* [308][**arXiv, 2021**] Habitat-Matterport 3D dataset (HM3D): 1000 large-scale 3d environments for embodied AI [paper](XXX)

#### 4.2.3 Medical Question Answering Datasets

* [309][**IJCARS, 2024**] Advancing surgical VQA with scene graph knowledge [paper](XXX)
* [310][**arXiv, 2024**] ErVQA: A dataset to benchmark the readiness of large vision language models in hospital environments [paper](XXX)
* [311][**arXiv, 2025**] MedReason: Eliciting factual medical reasoning steps in LLMs via knowledge graphs [paper](XXX)
* [312][**EMNLP, 2025**] ReasonMed: A 370k multi-agent generated dataset for advancing medical reasoning [paper](XXX)
* [313][**arXiv, 2025**] ORQA: A benchmark and foundation model for holistic operating room modeling [paper](XXX)
* [314][**arXiv, 2024**] LLaVA-Surg: towards multimodal surgical assistant via structured surgical video learning [paper](XXX)

### 4.3 Action Datasets

* [315][**MICCAI Workshop, 2014**] JHU-ISI gesture and skill assessment working set (JIGSAWS): A surgical activity dataset for human motion modeling [paper](XXX)
* [316][**arXiv, 2023**] The EndoScapes dataset for surgical scene segmentation, object detection, and critical view of safety assessment [paper](XXX)
* [317][**IJCARS, 2024**] Challenges in multi-centric generalization: phase and step recognition in Roux-en-Y gastric bypass surgery [paper](XXX)
* [318][**arXiv, 2025**] Surgical visual understanding (SurgVU) dataset [paper](XXX)
* [319][**arXiv, 2016**] The TUM LapChole dataset for the M2CAI 2016 workflow challenge [paper](XXX)
* [320][**arXiv, 2024**] General surgery vision transformer: A video pre-trained foundation model for general surgery [paper](XXX)
* [321][**arXiv, 2022**] MITI: SLAM benchmark for laparoscopic surgery [paper](XXX)
* [322][**ACM Multimedia, 2025**] COPESD: A multi-level surgical motion dataset for training large vision-language models to co-pilot endoscopic submucosal dissection [paper](XXX)

### 4.4 Simulation Platforms and Synthetic Datasets

#### 4.4.1 Surgical Simulation Platforms

* [323][**IROS, 2021**] SurRoL: An open-source reinforcement learning centered and dVRK compatible platform for surgical robot learning [paper](XXX)
* [324][**ICRA, 2024**] Orbit-Surgical: An open-simulation framework for learning surgical augmented dexterity [paper](XXX)
* [325][**ICRA, 2024**] Surgical Gym: A high-performance GPU-based platform for reinforcement learning with surgical robots [paper](XXX)
* [326][**Journal of Machine Learning Research, 2023**] LapGym-an open source framework for reinforcement learning in robot-assisted laparoscopic surgery [paper](XXX)
* [327][**arXiv, 2025**] SonoGym: High performance simulation for challenging surgical tasks with robotic ultrasound [paper](XXX)

#### 4.4.2 Synthetic Datasets

* [328][**arXiv, 2023**] SynFundus-1M: a high-quality million-scale synthetic fundus images dataset with fifteen types of annotation [paper](XXX)
* [329][**Scientific Data, 2023**] A large-scale synthetic pathological dataset for deep learning-enabled segmentation of breast cancer [paper](XXX)
* [330][**Intelligence-Based Medicine, 2020**] Synthea™ novel coronavirus (COVID-19) model and synthetic data set [paper](XXX)
* [331][**Electronics, 2022**] The “coherent data set”: combining patient data and imaging in a comprehensive, synthetic health record [paper](XXX)

---

<div align="center">
<a href="#top">⬆ Back to top</a>
</div>

