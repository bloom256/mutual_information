#include "mutual-information.h"

namespace Detail
{

class Histogram
{
public:
    Histogram(int bins);

    cv::Mat jointHist;
    cv::Mat refcHist;
    cv::Mat grayHist;
    int count = 0;
    int gray_sum = 0;
    int refc_sum = 0;
    int bins;

    float jointHistMaxVal = 0;
    float refcHistMaxVal = 0;
    float grayHistMaxVal = 0;

    float * jointHistData;
    float * refcHistData;
    float * grayHistData;
};

class Probability
{
public:
    Probability(int bins);

    // joint Probability
    cv::Mat jointProb;
    // marginal probability reflectivity
    cv::Mat refcProb;
    // marginal probability grayscale
    cv::Mat grayProb;
    int count = 0;
    int bins;

    float * jointProbData;
    float * refcProbData;
    float * grayProbData;
};

Histogram::Histogram(int bins) : bins(bins)
{
    jointHist = cv::Mat::zeros(bins, bins, CV_32FC1);
    refcHist = cv::Mat::zeros(1, bins, CV_32FC1);
    grayHist = cv::Mat::zeros(1, bins, CV_32FC1);

    jointHistData = (float *) jointHist.data;
    refcHistData = (float *) refcHist.data;
    grayHistData = (float *) grayHist.data;
}

Probability::Probability(int bins) : bins(bins)
{
    jointProb = cv::Mat::zeros(bins, bins, CV_32FC1);
    refcProb = cv::Mat::zeros(1, bins, CV_32FC1);
    grayProb = cv::Mat::zeros(1, bins, CV_32FC1);

    jointProbData = (float *) jointProb.data;
    refcProbData = (float *) refcProb.data;
    grayProbData = (float *) grayProb.data;
}

int RoundToInt(double value)
{
    return value < 0.0 ? int(value - 0.5) : int(value + 0.5);
}

Probability CalculateProbabilityMLE(Histogram hist)
{
    // Calculate sample covariance matrix
    float mu_gray = hist.gray_sum / hist.count;
    float mu_refc = hist.refc_sum / hist.count;

    // Covariances
    double sigma_gray = 0;
    double sigma_refc = 0;
    // Cross correlation
    double sigma_gr = 0;

    Probability probMLE(hist.bins);

    for (int i = 0; i < hist.bins; i++)
    {
        for (int j = 0; j < hist.bins; j++)
        {
            // Cross Correlation term;
            sigma_gr = sigma_gr + hist.jointHistData[i * hist.bins + j] * (i - mu_refc) * (j - mu_gray);
            // Normalize the histogram so that the value is between (0,1)
            probMLE.jointProbData[i * probMLE.bins + j] = hist.jointHistData[i * hist.bins + j] / (hist.count);
        }

        // calculate sample covariance
        sigma_gray = sigma_gray + (hist.grayHistData[i] * (i - mu_gray) * (i - mu_gray));
        sigma_refc = sigma_refc + (hist.refcHistData[i] * (i - mu_refc) * (i - mu_refc));

        probMLE.grayProbData[i] = hist.grayHistData[i] / hist.count;
        probMLE.refcProbData[i] = hist.refcHistData[i] / hist.count;
    }

    sigma_gray = sigma_gray / hist.count;
    sigma_refc = sigma_refc / hist.count;
    sigma_gr = sigma_gr / hist.count;

    // Compute the optimal bandwidth (Silverman's rule of thumb)
    sigma_gray = 1.06 * sqrt(sigma_gray) / pow(hist.count, 0.2);
    sigma_refc = 1.06 * sqrt(sigma_refc) / pow(hist.count, 0.2);

    // std::cout << "mu_gray " << mu_gray << " sigma_gray " << sigma_gray << std::endl;
    // std::cout << "mu_refc " << mu_refc << " sigma_refc " << sigma_refc << std::endl;

    cv::GaussianBlur(probMLE.grayProb, probMLE.grayProb, cv::Size(0, 0), sigma_gray);
    cv::GaussianBlur(probMLE.refcProb, probMLE.refcProb, cv::Size(0, 0), sigma_refc);
    cv::GaussianBlur(probMLE.jointProb, probMLE.jointProb, cv::Size(0, 0), sigma_gray, sigma_refc);
    probMLE.count = hist.count;
    return probMLE;
}

Histogram BuildHistogram(const std::vector<uint8_t> & grayValues,
                         const std::vector<uint8_t> & reflectanceValues,
                         int bins)
{
    if (grayValues.size() != reflectanceValues.size())
        throw std::runtime_error("MI::BuildHistogram error: the sizes of the input vectors are not equal");

    Histogram hist(bins);
    double binFraction = 256.0 / bins;
    for (size_t i = 0; i < grayValues.size(); i++)
    {
        int gray = int(grayValues[i] / binFraction);
        int refc = int(reflectanceValues[i] / binFraction);

        if (gray >= bins || refc >= bins || gray < 0 || refc < 0)
        {
            std::cout << gray << " " << refc << std::endl;
            throw std::runtime_error("index out of range");
        }
        // Update histograms
        hist.grayHistData[gray] = hist.grayHistData[gray] + 1;
        hist.refcHistData[refc] = hist.refcHistData[refc] + 1;
        hist.jointHistData[gray * bins + refc] = hist.jointHistData[gray * bins + refc] + 1;

        hist.count++;
        hist.gray_sum = hist.gray_sum + gray;
        hist.refc_sum = hist.refc_sum + refc;

        hist.jointHistMaxVal = std::max(hist.jointHistMaxVal, hist.jointHistData[gray * bins + refc]);
        hist.refcHistMaxVal = std::max(hist.refcHistMaxVal, hist.refcHistData[refc]);
        hist.grayHistMaxVal = std::max(hist.grayHistMaxVal, hist.grayHistData[gray]);
    }
    return hist;
}

void MakeDebugHistogramImages(Histogram hist, cv::Mat & grayHistImg, cv::Mat & refcHistImg, cv::Mat & jointHistImg)
{
    std::cout << "Histogram."
              << " count " << hist.count << " gray_sum " << hist.gray_sum << " refc_sum " << hist.refc_sum << " bins "
              << hist.bins << " jointHistMaxVal " << hist.jointHistMaxVal << " refcHistMaxVal " << hist.refcHistMaxVal
              << " grayHistMaxVal " << hist.grayHistMaxVal << std::endl;

    jointHistImg = cv::Mat::zeros(hist.bins, hist.bins, CV_8UC1);
    refcHistImg = cv::Mat::zeros(hist.bins, hist.bins, CV_8UC1);
    grayHistImg = cv::Mat::zeros(hist.bins, hist.bins, CV_8UC1);

    for (int i = 0; i < hist.bins; i++)
    {
        for (int j = 0; j < hist.bins; j++)
        {
            jointHistImg.at<uchar>(i, j) =
                RoundToInt(255.0 * hist.jointHistData[i * hist.bins + j] / hist.jointHistMaxVal);
        }

        int refcVal = RoundToInt(1.0 * (hist.bins - 1) * hist.refcHistData[i] / hist.refcHistMaxVal);
        for (int j = hist.bins - 1 - refcVal; j < hist.bins; j++)
        {
            refcHistImg.at<uchar>(j, i) = 255;
        }

        int grayVal = RoundToInt(1.0 * (hist.bins - 1) * hist.grayHistData[i] / hist.grayHistMaxVal);
        for (int j = hist.bins - 1 - grayVal; j < hist.bins; j++)
        {
            grayHistImg.at<uchar>(j, i) = 255;
        }
    }
}

void MakeDebugProbabiltyImages(Probability prob, cv::Mat & grayProbImg, cv::Mat & refcProbImg, cv::Mat & jointProbImg)
{
    std::cout << "Probability."
              << " count " << prob.count << " bins " << prob.bins << std::endl;

    jointProbImg = cv::Mat::zeros(prob.bins, prob.bins, CV_8UC1);
    refcProbImg = cv::Mat::zeros(prob.bins, prob.bins, CV_8UC1);
    grayProbImg = cv::Mat::zeros(prob.bins, prob.bins, CV_8UC1);

    float jointProbMaxVal = 0;
    float refcProbMaxVal = 0;
    float grayProbMaxVal = 0;

    for (int i = 0; i < prob.bins; i++)
    {
        for (int j = 0; j < prob.bins; j++)
        {
            jointProbMaxVal = std::max(jointProbMaxVal, prob.jointProbData[i * prob.bins + j]);
        }

        refcProbMaxVal = std::max(refcProbMaxVal, prob.refcProbData[i]);
        grayProbMaxVal = std::max(grayProbMaxVal, prob.grayProbData[i]);
    }

    std::cout << "jointProbMaxVal " << jointProbMaxVal << " refcProbMaxVal " << refcProbMaxVal << " grayProbMaxVal "
              << grayProbMaxVal << std::endl;

    for (int i = 0; i < prob.bins; i++)
    {
        for (int j = 0; j < prob.bins; j++)
        {
            jointProbImg.at<uchar>(i, j) = RoundToInt(255.0 * prob.jointProbData[i * prob.bins + j] / jointProbMaxVal);
        }

        int refcVal = RoundToInt(1.0 * (prob.bins - 1) * prob.refcProbData[i] / refcProbMaxVal);
        for (int j = prob.bins - 1 - refcVal; j < prob.bins; j++)
        {
            refcProbImg.at<uchar>(j, i) = 255;
        }

        int grayVal = RoundToInt(1.0 * (prob.bins - 1) * prob.grayProbData[i] / grayProbMaxVal);
        for (int j = prob.bins - 1 - grayVal; j < prob.bins; j++)
        {
            grayProbImg.at<uchar>(j, i) = 255;
        }
    }
}

void DrawDebugImages(const cv::Mat & grayHistImg,
                     const cv::Mat & refcHistImg,
                     const cv::Mat & jointHistImg,
                     const cv::Mat & grayProbImg,
                     const cv::Mat & refcProbImg,
                     const cv::Mat & jointProbImg)
{
    const std::string jointHistImgWin = "jointHistImg";
    cv::namedWindow(jointHistImgWin, cv::WINDOW_NORMAL);
    cv::imshow(jointHistImgWin, jointHistImg);

    const std::string refcHistImgWin = "refcHistImg";
    cv::namedWindow(refcHistImgWin, cv::WINDOW_NORMAL);
    cv::imshow(refcHistImgWin, refcHistImg);

    const std::string grayHistImgWin = "grayHistImg";
    cv::namedWindow(grayHistImgWin, cv::WINDOW_NORMAL);
    cv::imshow(grayHistImgWin, grayHistImg);

    const std::string jointProbImgWin = "jointProbImg";
    cv::namedWindow(jointProbImgWin, cv::WINDOW_NORMAL);
    cv::imshow(jointProbImgWin, jointProbImg);

    const std::string refcProbImgWin = "refcProbImg";
    cv::namedWindow(refcProbImgWin, cv::WINDOW_NORMAL);
    cv::imshow(refcProbImgWin, refcProbImg);

    const std::string grayProbImgWin = "grayProbImg";
    cv::namedWindow(grayProbImgWin, cv::WINDOW_NORMAL);
    cv::imshow(grayProbImgWin, grayProbImg);

    cv::waitKey(0);
    cv::destroyWindow(jointHistImgWin);
    cv::destroyWindow(refcHistImgWin);
    cv::destroyWindow(grayHistImgWin);
    cv::destroyWindow(jointProbImgWin);
    cv::destroyWindow(refcProbImgWin);
    cv::destroyWindow(grayProbImgWin);
}

void SaveDebugImages(const cv::Mat & grayHistImg,
                     const cv::Mat & refcHistImg,
                     const cv::Mat & jointHistImg,
                     const cv::Mat & grayProbImg,
                     const cv::Mat & refcProbImg,
                     const cv::Mat & jointProbImg,
                     const std::string & debugImagesPrefixPath)
{
    cv::imwrite(debugImagesPrefixPath + "_jointHist.png", jointHistImg);
    cv::imwrite(debugImagesPrefixPath + "_refcHist.png", refcHistImg);
    cv::imwrite(debugImagesPrefixPath + "_grayHist.png", grayHistImg);

    cv::imwrite(debugImagesPrefixPath + "_jointProb.png", jointProbImg);
    cv::imwrite(debugImagesPrefixPath + "_refcProb.png", refcProbImg);
    cv::imwrite(debugImagesPrefixPath + "_grayProb.png", grayProbImg);
}

} // namespace Detail

namespace MI
{

double Cost(const std::vector<uint8_t> & grayValues,
            const std::vector<uint8_t> & reflectanceValues,
            int bins,
            bool draw,
            std::string debugImagesPrefixPath)
{
    bins = std::min(256, bins);
    bins = std::max(2, bins);

    // Get MLE of probability distribution
    Detail::Histogram hist = Detail::BuildHistogram(grayValues, reflectanceValues, bins);
    Detail::Probability prob = Detail::CalculateProbabilityMLE(hist);

    if (draw || debugImagesPrefixPath != "")
    {
        cv::Mat grayHistImg, refcHistImg, jointHistImg;
        Detail::MakeDebugHistogramImages(hist, grayHistImg, refcHistImg, jointHistImg);
        cv::Mat grayProbImg, refcProbImg, jointProbImg;
        Detail::MakeDebugProbabiltyImages(prob, grayProbImg, refcProbImg, jointProbImg);

        if (debugImagesPrefixPath != "")
        {
            Detail::SaveDebugImages(grayHistImg, refcHistImg, jointHistImg, grayProbImg, refcProbImg, jointProbImg,
                                    debugImagesPrefixPath);
        }
        if (draw)
        {
            Detail::DrawDebugImages(grayHistImg, refcHistImg, jointHistImg, grayProbImg, refcProbImg, jointProbImg);
        }
    }

    double cost = 0;
    for (int i = 0; i < bins; i++)
    {
        for (int j = 0; j < bins; j++)
        {
            double pxy = prob.jointProbData[i * prob.bins + j];
            double px = prob.grayProbData[i];
            double py = prob.refcProbData[j];
            if (pxy > 0 && px > 0 && py > 0)
            {
                cost += pxy * std::log(pxy / px / py);
            }
        }
    }
    return cost;
}

} // namespace MI
