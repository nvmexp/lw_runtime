#ifndef GOLDEN_VALUE_CALLWLATOR_H
#define GOLDEN_VALUE_CALLWLATOR_H

#include <string>
#include <map>
#include <vector>

#include "Plugin.h"
#include "dcgm_structs.h"

typedef struct
{
    double value;     //!< The golden value (mean value) callwlated
    double variance;  //!< The variance callwlated for this mean in the dataset
} valueWithVariance_t;

typedef struct
{
    unsigned int        gpuId;           //!< The GPU id that these values were callwlated for
    valueWithVariance_t callwlatedValue; //!< The value and its variance
} dcgmGpuToValue_t;

typedef std::map<std::string, std::map<unsigned int, std::vector<double> > > paramToGpuToValues_t;
typedef std::map<std::string, std::vector<double> > paramToValues_t;

class GoldelwalueCallwlator
{
public:
    /*************************************************************************/
    GoldelwalueCallwlator() : m_inputs(), m_inputsPerGpu(), m_averageGpuValues(), m_callwlatedGoldelwalues()
    {
    }

    /*************************************************************************/
    /*
     * Store the values observed from the test in m_inputs and m_inputsPerGpu
     */
    void RecordGoldelwalueInputs(const std::string &testName, const observedMetrics_t &metrics);

    /*************************************************************************/
    /*
     * Callwlate and write the golden values and store them in m_callwlatedGoldelwalues using the inputs
     * recorded to this point
     *
     * Return DCGM_ST_VARIANCE if the values do not colwerge
     *        DCGM_ST_* on other errors
     *        DCGM_ST_OK on success
     */
    dcgmReturn_t CallwlateAndWriteGoldelwalues(const std::string &filename);
private:
    // "testname.parameter" -> array of values
    std::map<std::string, paramToValues_t> m_inputs;
    // "testname.parameter" -> gpuId -> array of values
    std::map<std::string, paramToGpuToValues_t> m_inputsPerGpu;
    // "testname.parameter" -> gpuId -> average value
    std::map<std::string, std::map<std::string, dcgmGpuToValue_t> > m_averageGpuValues;
    // "testname.parameter" -> average value
    std::map<std::string, std::map<std::string, valueWithVariance_t> > m_callwlatedGoldelwalues;

    // Methods
    /*************************************************************************/
    /*
     * Writes the configure file to the specified filename.
     */
    void WriteConfigFile(const std::string &filename);

    /*************************************************************************/
    /*
     * Returns the correct parameter name from the supplied metric name and test name
     * In many cases, it returns a string equal to metricName
     */
    std::string GetParameterName(const std::string &testname, const std::string &metricName) const;

    /*************************************************************************/
    /*
     * Records the value for the given parameter in our map of inputs
     */
    void AddToAllInputs(const std::string &testname, const std::string &paramName, double value);

    /*************************************************************************/
    /*
     * Records the value for the given parameter in our map of inputs which records gpu Id
     */
    void AddToInputsPerGpu(const std::string &testname, const std::string &paramName, unsigned int gpuId,
                           double value);

    /*************************************************************************/
    /*
     * Returns a callwlated mean and variance for the data in the vector
     */
    valueWithVariance_t CallwlateMeanAndVariance(std::vector<double> &data) const;

    /*************************************************************************/
    /*
     * Determines if this parameter should be adjusted up, down, or not at all
     * Returns a positive value to adjust up, 0 for not at all, and a negative value for down
     */
    double ToleranceAdjustFactor(const std::string &paramName) const;

    /*************************************************************************/
    /*
     * Adjusts the callwlated golden value for the testname and parameter to allow some
     * tolerance in the test runs.
     */
    void AdjustGoldelwalue(const std::string &testname, const std::string &paramName);

    /*************************************************************************/
    /*
     * Adjusts the callwlated golden value for the testname and parameter to allow some
     * tolerance in the test runs.
     */
    dcgmReturn_t IsVarianceAcceptable(const std::string &testname, const std::string &paramName,
                                      const valueWithVariance_t &vwv) const;

    /*************************************************************************/
    /*
     * Dumps all of the metrics we've recorded into a CSV file for inspection
     */
    void DumpObservedMetrics() const;

    /*************************************************************************/
    /*
     * Dumps all of the metrics we've recorded, but dumps the information per GPU
     */
    void DumpObservedMetricsWithGpuIds(int64_t timestamp) const;

    /*************************************************************************/
    /*
     * Dumps all of the metrics we've recorded with no information about which GPU it came from
     */
    void DumpObservedMetricsAll(int64_t timestamp) const;
};

#endif

