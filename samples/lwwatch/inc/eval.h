#ifndef __LWWATCH_EVAL_H
#define __LWWATCH_EVAL_H

class evaluator {
    const char * ibuf;
    unsigned * values;

    void skipwhite() {
        while (true) {
            if (*ibuf == ' ' || *ibuf == '\t')
                ibuf++;
            else if (*ibuf =='/' && ibuf[1] == '*') {
                ibuf+=2;
                while (*ibuf && *ibuf!='*' && ibuf[1]!='/')
                    ibuf++;
                if (*ibuf)
                    ibuf+=2;
            }
            else
                break;
        }
    }

    void match_token(string & tok) {
        skipwhite();
        tok = "";
        while ((*ibuf >= 'A' && *ibuf <= 'Z') || (*ibuf >= 'a' && *ibuf <= 'z') || *ibuf=='_' || (*ibuf >= '0' && *ibuf <= '9') )
            tok+=*ibuf++;
    }

    bool eval_term(unsigned & result) {
        skipwhite();
        if (*ibuf >= '0' && *ibuf <= '9') 
        {
            char *endptr = 0;
            result = strtoul(ibuf, &endptr, 0);
            ibuf = endptr;
            return true;
        }
        else if ((*ibuf >= 'A' && *ibuf <= 'Z') ||
                 (*ibuf >= 'a' && *ibuf <= 'z')) {
            string word;
            match_token(word);

            if (parameters.find(word)==parameters.end())
                return false;

            if (!values) {                
                return false;   // WARNING
            }
            result = values[parameters[word]];
        }
        else if (*ibuf == '(') {
            ibuf++;
            if (!eval_ternary(result))
                return false;
            skipwhite();
            if (*ibuf != ')')
                return false;
            ibuf++;
        } 
        else
            return false;
        return true;
    }

    bool eval_factor(unsigned & result) {
        if (!eval_term(result))
            return false;

        while (1) {
            skipwhite();
            if (*ibuf == '*' || *ibuf == '/' || *ibuf == '%') {
                char ch = *ibuf++;
                unsigned right;
                if (!eval_term(right))
                    return false;

                if (ch == '*')
                    result *= right;
                else if (ch == '/')
                    result /= right;
                else
                    result %= right;
            }
            else
                break;
        }
        return true;
    }
    
   
    bool eval_add_subtr(unsigned & result) 
    {
        if (!eval_factor(result))
            return false;

        while (1) {
            skipwhite();
            if (*ibuf == '+' || *ibuf == '-') {
                char ch = *ibuf++;
                unsigned right;
                if (!eval_factor(right))
                    return false;

                if (ch == '+')
                    result += right;
                else
                    result -= right;
            }
            else
                break;
        }
        return true;
    }

    bool eval_shift(unsigned &result)
    {
        if (!eval_add_subtr(result))
            return false;
        while (1) {
            skipwhite();
            if ((*ibuf == '<' && *(ibuf+1) == '<') || (*ibuf == '>' && *(ibuf+1) == '>')) {
                char ch = *ibuf++;
                ibuf++;
                unsigned right;
                if (!eval_add_subtr(right))
                    return false;

                if (ch == '<')
                    result = (result << right);
                else
                    result = (result >> right);
            }
            else
                break;
        }
        return true;
    }

    bool eval_comp_less_gr(unsigned &result)
    {
        if (!eval_shift(result))
            return false;
        while (1) {
            skipwhite();
            if ((*ibuf == '<' && *(ibuf+1) != '<') || (*ibuf == '<' && *(ibuf+1) == '=') || 
                (*ibuf == '>' && *(ibuf+1) != '>') || (*ibuf == '>' && *(ibuf+1) == '=')) {
                char ch[2] = { '\0', '\0' };
                ch[0] = *ibuf++;
                if (*ibuf == '=') {
                    ch[1] = *ibuf++;
                }
                unsigned right;
                if (!eval_shift(right))
                    return false;

                if (ch[1] == '=') {
                    if (ch[0] == '<')
                        result = (result <= right);
                    else
                        result = (result >= right);
                }
                else {
                    if (ch[0] == '<')
                        result = (result < right);
                    else
                        result = (result > right);
                }
            }
            else
                break;
        }
        return true;
    }

    bool eval_comp_equal(unsigned &result)
    {
        if (!eval_comp_less_gr(result))
            return false;
        while (1) {
            skipwhite();
            if ((*ibuf == '=' && *(ibuf+1) == '=') || (*ibuf == '!' && *(ibuf+1) == '=')) {
                char ch = *ibuf++;
                ibuf++;
                unsigned right;
                if (!eval_comp_less_gr(right))
                    return false;

                if (ch == '=') {
                    if (result == right)
                        result = 1;
                    else
                        result = 0;
                }
                else {
                    if (result == right)
                        result = 0;
                    else
                        result = 1;
                }
            }
            else
                break;
        }
        return true;
    }
 
    bool eval_bit_and(unsigned &result)
    {
        if (!eval_comp_equal(result))
            return false;
        while (1) {
            skipwhite();
            if (*ibuf == '&' && *(ibuf+1) != '&') {
                ibuf++;
                unsigned right;
                if (!eval_comp_equal(right))
                    return false;

                result &= right;
            }
            else
                break;
        }
        return true;
    }
    
    bool eval_bit_extern_or(unsigned &result)
    {
        if (!eval_bit_and(result))
            return false;
        while (1) {
            skipwhite();
            if (*ibuf == '^') {
                ibuf++;
                unsigned right;
                if (!eval_bit_and(right))
                    return false;

                result ^= right;
            }
            else
                break;
        }
        return true;
    }

    bool eval_bit_intern_or(unsigned &result)
    {
        if (!eval_bit_extern_or(result))
            return false;
        while (1) {
            skipwhite();
            if (*ibuf == '|' && *(ibuf+1) != '|') {
                ibuf++;
                unsigned right;
                if (!eval_bit_extern_or(right))
                    return false;

                result |= right;
            }
            else
                break;
        }
        return true;
    }

    bool eval_logic_and(unsigned &result)
    {
        if (!eval_bit_intern_or(result))
            return false;
        while (1) {
            skipwhite();
            if ((*ibuf == '&' && *(ibuf+1) == '&')) {
                ibuf++;
                ibuf++;
                unsigned right;
                if (!eval_bit_intern_or(right))
                    return false;

                result = (result && right);
            }
            else
                break;
        }
        return true;
    }
 
    bool eval_logic_or(unsigned &result)
    {
        if (!eval_logic_and(result))
            return false;
        while (1) {
            skipwhite();
            if ((*ibuf == '|' && *(ibuf+1) == '|')) {
                ibuf++;
                ibuf++;
                unsigned right;
                if (!eval_logic_and(right))
                    return false;

                result = (result || right);
            }
            else
                break;
        }
        return true;
    }

    bool eval_ternary_suffix(unsigned & result)
    {
        unsigned if_true;
        if (!eval_ternary(if_true))
            return false;

        skipwhite();
        if (*ibuf!=':')
            return false;
        ibuf++;

        unsigned if_false;
        if (!eval_ternary(if_false))
            return false;

        // analyze condition
        if (result)
            result = if_true;
        else
            result = if_false;

        return true;
    }

    bool eval_ternary(unsigned & result) {
        if (!eval_logic_or(result))
            return false;

        skipwhite();

        if (*ibuf=='?') {
            ibuf++;
            return eval_ternary_suffix(result);
        }
        return true;
    }

public:
    map<string, unsigned> parameters;
    string expr;


    bool eval_range(unsigned & if_true, unsigned & if_false, unsigned * arguments) {
        ibuf = expr.c_str();
        values = arguments;

        if_false = 0;
        if (!eval_ternary_suffix(if_false))
            return false;

        ibuf = expr.c_str();
        if_true = 1;
        if (!eval_ternary_suffix(if_true))
            return false;

        return true;
   
    }

    bool eval(unsigned & result, unsigned * arguments)
    {
        ibuf = expr.c_str();
        values = arguments;
        if (!eval_ternary(result))
            return false;
        return true;
    }

    // 
    // The public interface for using the function
    // "eval_expression" from outside. 
    // 
    bool eval_expr(unsigned &result, const char * inputExpression)
    {
        ibuf = inputExpression;
        //return true only when there are no stray character to the end of ibuf.. 
        return (eval_logic_or(result) && (*ibuf) == '\0') ;
    }
};

#endif
