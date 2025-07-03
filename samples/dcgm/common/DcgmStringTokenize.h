#ifndef DCGMSTRINGTOKENIZE_H
#define DCGMSTRINGTOKENIZE_H

#include <string>
#include <vector>

static
void tokenizeString(const std::string &src, const std::string &delimiter, std::vector<std::string> &tokens)
{
    size_t pos = 0;
    size_t prev_pos = 0;

    if (src.size() > 0) {
        while (pos != std::string::npos) {
            std::string token;
            pos = src.find(delimiter, prev_pos);

            if (pos == std::string::npos) {
                token = src.substr(prev_pos);
            } else {
                token = src.substr(prev_pos, pos - prev_pos);
                prev_pos = pos + delimiter.size();
            }

            tokens.push_back(token);
        }
    }
}

static
std::vector<std::string> tokenizeString(const std::string &src, const std::string &delimiter)
{
    std::vector<std::string> tokens;

    tokenizeString(src, delimiter, tokens);

    return(tokens);
}

#endif // DCGMSTRINGTOKENIZE_H
