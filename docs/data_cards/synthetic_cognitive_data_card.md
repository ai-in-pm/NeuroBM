# Data Card: Synthetic Cognitive Features Dataset

## Motivation

### Purpose
The synthetic cognitive features dataset was created to enable research and education in cognitive modeling while avoiding privacy concerns associated with real human data. This dataset supports the development and testing of Boltzmann machine models for understanding cognitive patterns.

### Creators
- **Organization**: NeuroBM Development Team
- **Funding**: Research and educational initiative
- **Contact**: [contact information]

### Use Cases
1. **Educational Training**: Teaching cognitive modeling concepts
2. **Algorithm Development**: Testing new machine learning algorithms
3. **Baseline Research**: Providing controlled baselines for cognitive studies
4. **Methodology Validation**: Validating statistical and computational methods

## Composition

### Dataset Description
- **Data Type**: Synthetic cognitive feature vectors
- **Format**: Continuous values in [0,1] range
- **Size**: Configurable (typically 1,000-10,000 samples)
- **Dimensionality**: 5-6 features per cognitive regime
- **File Format**: CSV, NumPy arrays, PyTorch tensors

### Feature Definitions

#### Base Cognitive Regime
1. **Attention Span** (0-1): Ability to maintain focused attention
2. **Working Memory Proxy** (0-1): Capacity for temporary information storage
3. **Novelty Seeking** (0-1): Tendency to seek new experiences
4. **Sleep Quality** (0-1): Overall sleep quality and patterns
5. **Stress Index** (0-1): General stress level indicator

#### PTSD-Related Regime
1. **Hyperarousal Proxy** (0-1): Physiological and psychological arousal
2. **Startle Sensitivity** (0-1): Exaggerated startle response
3. **Avoidance Tendency** (0-1): Behavioral and cognitive avoidance
4. **Intrusive Thought Proxy** (0-1): Frequency of intrusive thoughts
5. **Sleep Disruption** (0-1): Sleep pattern disruption
6. **Threat Bias Proxy** (0-1): Attentional bias toward threats

### Data Instances
- **Sample Count**: Variable (1K-10K typical)
- **Missing Values**: None (synthetic generation ensures completeness)
- **Duplicates**: None (each sample is unique)
- **Outliers**: Controlled through generation parameters

## Collection Process

### Data Generation Method
- **Algorithm**: Multivariate normal with controlled correlations
- **Correlation Structure**: Based on literature review of cognitive relationships
- **Distribution Shaping**: Skewed distributions to match realistic patterns
- **Noise Addition**: Controlled noise to prevent overfitting

### Generation Parameters
- **Random Seed**: Configurable for reproducibility
- **Correlation Matrix**: Predefined based on cognitive literature
- **Skewness Parameters**: Adjusted per feature for realism
- **Noise Level**: Typically 1-5% of signal

### Quality Control
- **Statistical Validation**: Verification of intended correlations
- **Range Checking**: Ensuring all values in [0,1] range
- **Distribution Analysis**: Verification of intended distributions
- **Correlation Verification**: Checking correlation matrix accuracy

## Preprocessing/Cleaning/Labeling

### Preprocessing Steps
1. **Generation**: Create raw multivariate samples
2. **Transformation**: Apply skewness and realistic distributions
3. **Normalization**: Scale to [0,1] range
4. **Validation**: Check statistical properties
5. **Export**: Save in multiple formats

### Data Transformations
- **Min-Max Scaling**: Normalize to [0,1] range
- **Clipping**: Ensure values stay within bounds
- **Correlation Adjustment**: Fine-tune correlations to target values
- **Distribution Shaping**: Apply realistic skewness patterns

### Labeling Process
- **Feature Names**: Descriptive names for each cognitive feature
- **Metadata**: Generation parameters and timestamps
- **Documentation**: Detailed description of each feature
- **Validation Flags**: Quality control indicators

## Uses

### Current Uses
1. **Research**: Cognitive modeling research and development
2. **Education**: Teaching machine learning and cognitive science
3. **Testing**: Algorithm validation and comparison
4. **Demonstration**: Showcasing Boltzmann machine capabilities

### Potential Future Uses
1. **Benchmark Development**: Creating standardized benchmarks
2. **Method Validation**: Testing new statistical methods
3. **Simulation Studies**: Large-scale simulation experiments
4. **Tool Development**: Building cognitive modeling tools

### Inappropriate Uses
- ❌ **Clinical Applications**: Never use for real clinical decisions
- ❌ **Individual Assessment**: Never use to assess real individuals
- ❌ **Diagnostic Tools**: Never use as basis for diagnostic instruments
- ❌ **Screening Applications**: Never use for screening or selection

## Distribution

### Distribution Method
- **Repository**: Available through NeuroBM GitHub repository
- **License**: MIT License (open source)
- **Access**: Free and open access
- **Format**: Multiple formats (CSV, NumPy, PyTorch)

### Version Control
- **Versioning**: Semantic versioning for dataset releases
- **Changelog**: Detailed changelog for each version
- **Reproducibility**: Fixed seeds for reproducible generation
- **Documentation**: Comprehensive documentation for each version

## Maintenance

### Maintenance Plan
- **Regular Updates**: Annual review and potential updates
- **Bug Fixes**: Prompt fixing of any generation issues
- **Feature Additions**: New cognitive regimes as needed
- **Documentation Updates**: Keep documentation current

### Maintenance Team
- **Primary Maintainers**: NeuroBM core development team
- **Domain Experts**: Cognitive science consultants
- **Community**: Open source community contributions
- **Quality Assurance**: Dedicated QA team

## Ethical Considerations

### Privacy
- **No Personal Data**: Entirely synthetic, no real individual data
- **No Identifiers**: No personally identifiable information
- **Anonymity**: Cannot be linked to real individuals
- **Consent**: No consent required (synthetic data)

### Bias and Fairness
- **Generation Bias**: Potential biases in correlation assumptions
- **Population Representation**: May not represent all populations equally
- **Cultural Bias**: Correlations based on Western research literature
- **Demographic Gaps**: Limited representation of diverse populations

### Potential Harms
- **Misinterpretation**: Risk of treating synthetic data as real
- **Oversimplification**: May oversimplify complex cognitive relationships
- **False Validation**: Good performance on synthetic data ≠ real-world validity
- **Bias Amplification**: May amplify existing research biases

### Mitigation Strategies
- **Clear Labeling**: All data clearly marked as synthetic
- **Documentation**: Extensive documentation of limitations
- **Education**: User education about appropriate use
- **Regular Review**: Regular ethical review of data generation

## Limitations

### Known Limitations
1. **Synthetic Nature**: Not derived from real human data
2. **Simplified Relationships**: Real cognitive relationships are more complex
3. **Static Correlations**: Correlations don't vary across individuals
4. **No Temporal Dynamics**: No modeling of temporal changes
5. **Limited Validation**: No validation against real cognitive assessments

### Statistical Limitations
- **Fixed Correlations**: Correlation structure is predetermined
- **Gaussian Assumptions**: Underlying generation assumes multivariate normal
- **Independence**: Samples are statistically independent
- **Stationarity**: No temporal or developmental changes

### Generalizability Concerns
- **Population Scope**: Based on general population assumptions
- **Cultural Limitations**: May not generalize across cultures
- **Age Limitations**: No age-related cognitive changes
- **Individual Differences**: Limited individual variation modeling

## Recommendations

### Best Practices
1. **Educational Use**: Ideal for teaching and learning
2. **Method Development**: Good for developing new methods
3. **Baseline Studies**: Useful for baseline comparisons
4. **Always Validate**: Validate any findings with real data
5. **Document Limitations**: Always document synthetic nature

### Validation Requirements
- **Real Data Comparison**: Compare with real cognitive data when possible
- **Cross-Validation**: Use proper cross-validation techniques
- **External Validation**: Seek external validation of findings
- **Replication**: Replicate findings across different synthetic datasets

## Contact Information

### Data Card Authors
- NeuroBM Development Team
- Cognitive Science Consultants
- Ethics Review Committee

### Contact Details
- **Email**: [contact information]
- **GitHub Issues**: [repository link]
- **Documentation**: [documentation link]
- **Community Forum**: [forum link]

---

**Important Notice**: This dataset is entirely synthetic and created for research and educational purposes only. It should never be used for clinical diagnosis, treatment decisions, or any application affecting real individuals without extensive validation against real-world data.

**Version**: 1.0  
**Last Updated**: 2024  
**Next Review**: Annual
