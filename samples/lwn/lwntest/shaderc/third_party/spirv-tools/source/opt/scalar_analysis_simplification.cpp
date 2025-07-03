// Copyright (c) 2018 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "source/opt/scalar_analysis.h"

#include <functional>
#include <map>
#include <memory>
#include <set>
#include <unordered_set>
#include <utility>
#include <vector>

// Simplifies scalar analysis DAGs.
//
// 1. Given a node passed to SimplifyExpression we first simplify the graph by
// calling SimplifyPolynomial. This groups like nodes following basic arithmetic
// rules, so multiple adds of the same load instruction could be grouped into a
// single multiply of that instruction. SimplifyPolynomial will traverse the DAG
// and build up an aclwmulator buffer for each class of instruction it finds.
// For example take the loop:
// for (i=0, i<N; i++) { i+B+23+4+B+C; }
// In this example the expression "i+B+23+4+B+C" has four classes of
// instruction, induction variable i, the two value unknowns B and C, and the
// constants. The aclwmulator buffer is then used to rebuild the graph using
// the aclwmulation of each type. This example would then be folded into
// i+2*B+C+27.
//
// This new graph contains a single add node (or if only one type found then
// just that node) with each of the like terms (or multiplication node) as a
// child.
//
// 2. FoldRelwrrentAddExpressions is then called on this new DAG. This will take
// RelwrrentAddExpressions which are with respect to the same loop and fold them
// into a single new RelwrrentAddExpression with respect to that same loop. An
// expression can have multiple RelwrrentAddExpression's with respect to
// different loops in the case of nested loops. These expressions cannot be
// folded further. For example:
//
// for (i=0; i<N;i++) for(j=0,k=1; j<N;++j,++k)
//
// The 'j' and 'k' are RelwrrentAddExpression with respect to the second loop
// and 'i' to the first. If 'j' and 'k' are used in an expression together then
// they will be folded into a new RelwrrentAddExpression with respect to the
// second loop in that expression.
//
//
// 3. If the DAG now only contains a single RelwrrentAddExpression we can now
// perform a final optimization SimplifyRelwrrentAddExpression. This will
// transform the entire DAG into a RelwrrentAddExpression. Additions to the
// RelwrrentAddExpression are added to the offset field and multiplications to
// the coefficient.
//

namespace spvtools {
namespace opt {

// Implementation of the functions which are used to simplify the graph. Graphs
// of unknowns, multiplies, additions, and constants can be turned into a linear
// add node with each term as a child. For instance a large graph built from, X
// + X*2 + Y - Y*3 + 4 - 1, would become a single add expression with the
// children X*3, -Y*2, and the constant 3. Graphs containing a relwrrent
// expression will be simplified to represent the entire graph around a single
// relwrrent expression. So for an induction variable (i=0, i++) if you add 1 to
// i in an expression we can rewrite the graph of that expression to be a single
// relwrrent expression of (i=1,i++).
class SENodeSimplifyImpl {
 public:
  SENodeSimplifyImpl(ScalarEvolutionAnalysis* analysis,
                     SENode* node_to_simplify)
      : analysis_(*analysis),
        node_(node_to_simplify),
        constant_aclwmulator_(0) {}

  // Return the result of the simplification.
  SENode* Simplify();

 private:
  // Relwrsively descend through the graph to build up the aclwmulator objects
  // which are used to flatten the graph. |child| is the node lwrrenty being
  // traversed and the |negation| flag is used to signify that this operation
  // was preceded by a unary negative operation and as such the result should be
  // negated.
  void GatherAclwmulatorsFromChildNodes(SENode* new_node, SENode* child,
                                        bool negation);

  // Given a |multiply| node add to the aclwmulators for the term type within
  // the |multiply| expression. Will return true if the aclwmulators could be
  // callwlated successfully. If the |multiply| is in any form other than
  // unknown*constant then we return false. |negation| signifies that the
  // operation was preceded by a unary negative.
  bool AclwmulatorsFromMultiply(SENode* multiply, bool negation);

  SERelwrrentNode* UpdateCoefficient(SERelwrrentNode* relwrrent,
                                     int64_t coefficient_update) const;

  // If the graph contains a relwrrent expression, ie, an expression with the
  // loop iterations as a term in the expression, then the whole expression
  // can be rewritten to be a relwrrent expression.
  SENode* SimplifyRelwrrentAddExpression(SERelwrrentNode* node);

  // Simplify the whole graph by linking like terms together in a single flat
  // add node. So X*2 + Y -Y + 3 +6 would become X*2 + 9. Where X and Y are a
  // ValueUnknown node (i.e, a load) or a relwrrent expression.
  SENode* SimplifyPolynomial();

  // Each relwrrent expression is an expression with respect to a specific loop.
  // If we have two different relwrrent terms with respect to the same loop in a
  // single expression then we can fold those terms into a single new term.
  // For instance:
  //
  // induction i = 0, i++
  // temp = i*10
  // array[i+temp]
  //
  // We can fold the i + temp into a single expression. Rec(0,1) + Rec(0,10) can
  // become Rec(0,11).
  SENode* FoldRelwrrentAddExpressions(SENode*);

  // We can eliminate relwrrent expressions which have a coefficient of zero by
  // replacing them with their offset value. We are able to do this because a
  // relwrrent expression represents the equation coefficient*iterations +
  // offset.
  SENode* EliminateZeroCoefficientRelwrrents(SENode* node);

  // A reference the the analysis which requested the simplification.
  ScalarEvolutionAnalysis& analysis_;

  // The node being simplified.
  SENode* node_;

  // An aclwmulator of the net result of all the constant operations performed
  // in a graph.
  int64_t constant_aclwmulator_;

  // An aclwmulator for each of the non constant terms in the graph.
  std::map<SENode*, int64_t> aclwmulators_;
};

// From a |multiply| build up the aclwmulator objects.
bool SENodeSimplifyImpl::AclwmulatorsFromMultiply(SENode* multiply,
                                                  bool negation) {
  if (multiply->GetChildren().size() != 2 ||
      multiply->GetType() != SENode::Multiply)
    return false;

  SENode* operand_1 = multiply->GetChild(0);
  SENode* operand_2 = multiply->GetChild(1);

  SENode* value_unknown = nullptr;
  SENode* constant = nullptr;

  // Work out which operand is the unknown value.
  if (operand_1->GetType() == SENode::ValueUnknown ||
      operand_1->GetType() == SENode::RelwrrentAddExpr)
    value_unknown = operand_1;
  else if (operand_2->GetType() == SENode::ValueUnknown ||
           operand_2->GetType() == SENode::RelwrrentAddExpr)
    value_unknown = operand_2;

  // Work out which operand is the constant coefficient.
  if (operand_1->GetType() == SENode::Constant)
    constant = operand_1;
  else if (operand_2->GetType() == SENode::Constant)
    constant = operand_2;

  // If the expression is not a variable multiplied by a constant coefficient,
  // exit out.
  if (!(value_unknown && constant)) {
    return false;
  }

  int64_t sign = negation ? -1 : 1;

  auto iterator = aclwmulators_.find(value_unknown);
  int64_t new_value = constant->AsSEConstantNode()->FoldToSingleValue() * sign;
  // Add the result of the multiplication to the aclwmulators.
  if (iterator != aclwmulators_.end()) {
    (*iterator).second += new_value;
  } else {
    aclwmulators_.insert({value_unknown, new_value});
  }

  return true;
}

SENode* SENodeSimplifyImpl::Simplify() {
  // We only handle graphs with an addition, multiplication, or negation, at the
  // root.
  if (node_->GetType() != SENode::Add && node_->GetType() != SENode::Multiply &&
      node_->GetType() != SENode::Negative)
    return node_;

  SENode* simplified_polynomial = SimplifyPolynomial();

  SERelwrrentNode* relwrrent_expr = nullptr;
  node_ = simplified_polynomial;

  // Fold relwrrent expressions which are with respect to the same loop into a
  // single relwrrent expression.
  simplified_polynomial = FoldRelwrrentAddExpressions(simplified_polynomial);

  simplified_polynomial =
      EliminateZeroCoefficientRelwrrents(simplified_polynomial);

  // Traverse the immediate children of the new node to find the relwrrent
  // expression. If there is more than one there is nothing further we can do.
  for (SENode* child : simplified_polynomial->GetChildren()) {
    if (child->GetType() == SENode::RelwrrentAddExpr) {
      relwrrent_expr = child->AsSERelwrrentNode();
    }
  }

  // We need to count the number of unique relwrrent expressions in the DAG to
  // ensure there is only one.
  for (auto child_iterator = simplified_polynomial->graph_begin();
       child_iterator != simplified_polynomial->graph_end(); ++child_iterator) {
    if (child_iterator->GetType() == SENode::RelwrrentAddExpr &&
        relwrrent_expr != child_iterator->AsSERelwrrentNode()) {
      return simplified_polynomial;
    }
  }

  if (relwrrent_expr) {
    return SimplifyRelwrrentAddExpression(relwrrent_expr);
  }

  return simplified_polynomial;
}

// Traverse the graph to build up the aclwmulator objects.
void SENodeSimplifyImpl::GatherAclwmulatorsFromChildNodes(SENode* new_node,
                                                          SENode* child,
                                                          bool negation) {
  int32_t sign = negation ? -1 : 1;

  if (child->GetType() == SENode::Constant) {
    // Collect all the constants and add them together.
    constant_aclwmulator_ +=
        child->AsSEConstantNode()->FoldToSingleValue() * sign;

  } else if (child->GetType() == SENode::ValueUnknown ||
             child->GetType() == SENode::RelwrrentAddExpr) {
    // To rebuild the graph of X+X+X*2 into 4*X we count the oclwrrences of X
    // and create a new node of count*X after. X can either be a ValueUnknown or
    // a RelwrrentAddExpr. The count for each X is stored in the aclwmulators_
    // map.

    auto iterator = aclwmulators_.find(child);
    // If we've encountered this term before add to the aclwmulator for it.
    if (iterator == aclwmulators_.end())
      aclwmulators_.insert({child, sign});
    else
      iterator->second += sign;

  } else if (child->GetType() == SENode::Multiply) {
    if (!AclwmulatorsFromMultiply(child, negation)) {
      new_node->AddChild(child);
    }

  } else if (child->GetType() == SENode::Add) {
    for (SENode* next_child : *child) {
      GatherAclwmulatorsFromChildNodes(new_node, next_child, negation);
    }

  } else if (child->GetType() == SENode::Negative) {
    SENode* negated_node = child->GetChild(0);
    GatherAclwmulatorsFromChildNodes(new_node, negated_node, !negation);
  } else {
    // If we can't work out how to fold the expression just add it back into
    // the graph.
    new_node->AddChild(child);
  }
}

SERelwrrentNode* SENodeSimplifyImpl::UpdateCoefficient(
    SERelwrrentNode* relwrrent, int64_t coefficient_update) const {
  std::unique_ptr<SERelwrrentNode> new_relwrrent_node{new SERelwrrentNode(
      relwrrent->GetParentAnalysis(), relwrrent->GetLoop())};

  SENode* new_coefficient = analysis_.CreateMultiplyNode(
      relwrrent->GetCoefficient(),
      analysis_.CreateConstant(coefficient_update));

  // See if the node can be simplified.
  SENode* simplified = analysis_.SimplifyExpression(new_coefficient);
  if (simplified->GetType() != SENode::CanNotCompute)
    new_coefficient = simplified;

  if (coefficient_update < 0) {
    new_relwrrent_node->AddOffset(
        analysis_.CreateNegation(relwrrent->GetOffset()));
  } else {
    new_relwrrent_node->AddOffset(relwrrent->GetOffset());
  }

  new_relwrrent_node->AddCoefficient(new_coefficient);

  return analysis_.GetCachedOrAdd(std::move(new_relwrrent_node))
      ->AsSERelwrrentNode();
}

// Simplify all the terms in the polynomial function.
SENode* SENodeSimplifyImpl::SimplifyPolynomial() {
  std::unique_ptr<SENode> new_add{new SEAddNode(node_->GetParentAnalysis())};

  // Traverse the graph and gather the aclwmulators from it.
  GatherAclwmulatorsFromChildNodes(new_add.get(), node_, false);

  // Fold all the constants into a single constant node.
  if (constant_aclwmulator_ != 0) {
    new_add->AddChild(analysis_.CreateConstant(constant_aclwmulator_));
  }

  for (auto& pair : aclwmulators_) {
    SENode* term = pair.first;
    int64_t count = pair.second;

    // We can eliminate the term completely.
    if (count == 0) continue;

    if (count == 1) {
      new_add->AddChild(term);
    } else if (count == -1 && term->GetType() != SENode::RelwrrentAddExpr) {
      // If the count is -1 we can just add a negative version of that node,
      // unless it is a relwrrent expression as we would rather the negative
      // goes on the relwrrent expressions children. This makes it easier to
      // work with in other places.
      new_add->AddChild(analysis_.CreateNegation(term));
    } else {
      // Output value unknown terms as count*term and output relwrrent
      // expression terms as rec(offset, coefficient + count) offset and
      // coefficient are the same as in the original expression.
      if (term->GetType() == SENode::ValueUnknown) {
        SENode* count_as_constant = analysis_.CreateConstant(count);
        new_add->AddChild(
            analysis_.CreateMultiplyNode(count_as_constant, term));
      } else {
        assert(term->GetType() == SENode::RelwrrentAddExpr &&
               "We only handle value unknowns or relwrrent expressions");

        // Create a new relwrrent expression by adding the count to the
        // coefficient of the old one.
        new_add->AddChild(UpdateCoefficient(term->AsSERelwrrentNode(), count));
      }
    }
  }

  // If there is only one term in the addition left just return that term.
  if (new_add->GetChildren().size() == 1) {
    return new_add->GetChild(0);
  }

  // If there are no terms left in the addition just return 0.
  if (new_add->GetChildren().size() == 0) {
    return analysis_.CreateConstant(0);
  }

  return analysis_.GetCachedOrAdd(std::move(new_add));
}

SENode* SENodeSimplifyImpl::FoldRelwrrentAddExpressions(SENode* root) {
  std::unique_ptr<SEAddNode> new_node{new SEAddNode(&analysis_)};

  // A mapping of loops to the list of relwrrent expressions which are with
  // respect to those loops.
  std::map<const Loop*, std::vector<std::pair<SERelwrrentNode*, bool>>>
      loops_to_relwrrent{};

  bool has_multiple_same_loop_relwrrent_terms = false;

  for (SENode* child : *root) {
    bool negation = false;

    if (child->GetType() == SENode::Negative) {
      child = child->GetChild(0);
      negation = true;
    }

    if (child->GetType() == SENode::RelwrrentAddExpr) {
      const Loop* loop = child->AsSERelwrrentNode()->GetLoop();

      SERelwrrentNode* rec = child->AsSERelwrrentNode();
      if (loops_to_relwrrent.find(loop) == loops_to_relwrrent.end()) {
        loops_to_relwrrent[loop] = {std::make_pair(rec, negation)};
      } else {
        loops_to_relwrrent[loop].push_back(std::make_pair(rec, negation));
        has_multiple_same_loop_relwrrent_terms = true;
      }
    } else {
      new_node->AddChild(child);
    }
  }

  if (!has_multiple_same_loop_relwrrent_terms) return root;

  for (auto pair : loops_to_relwrrent) {
    std::vector<std::pair<SERelwrrentNode*, bool>>& relwrrent_expressions =
        pair.second;
    const Loop* loop = pair.first;

    std::unique_ptr<SENode> new_coefficient{new SEAddNode(&analysis_)};
    std::unique_ptr<SENode> new_offset{new SEAddNode(&analysis_)};

    for (auto node_pair : relwrrent_expressions) {
      SERelwrrentNode* node = node_pair.first;
      bool negative = node_pair.second;

      if (!negative) {
        new_coefficient->AddChild(node->GetCoefficient());
        new_offset->AddChild(node->GetOffset());
      } else {
        new_coefficient->AddChild(
            analysis_.CreateNegation(node->GetCoefficient()));
        new_offset->AddChild(analysis_.CreateNegation(node->GetOffset()));
      }
    }

    std::unique_ptr<SERelwrrentNode> new_relwrrent{
        new SERelwrrentNode(&analysis_, loop)};

    SENode* new_coefficient_simplified =
        analysis_.SimplifyExpression(new_coefficient.get());

    SENode* new_offset_simplified =
        analysis_.SimplifyExpression(new_offset.get());

    if (new_coefficient_simplified->GetType() == SENode::Constant &&
        new_coefficient_simplified->AsSEConstantNode()->FoldToSingleValue() ==
            0) {
      return new_offset_simplified;
    }

    new_relwrrent->AddCoefficient(new_coefficient_simplified);
    new_relwrrent->AddOffset(new_offset_simplified);

    new_node->AddChild(analysis_.GetCachedOrAdd(std::move(new_relwrrent)));
  }

  // If we only have one child in the add just return that.
  if (new_node->GetChildren().size() == 1) {
    return new_node->GetChild(0);
  }

  return analysis_.GetCachedOrAdd(std::move(new_node));
}

SENode* SENodeSimplifyImpl::EliminateZeroCoefficientRelwrrents(SENode* node) {
  if (node->GetType() != SENode::Add) return node;

  bool has_change = false;

  std::vector<SENode*> new_children{};
  for (SENode* child : *node) {
    if (child->GetType() == SENode::RelwrrentAddExpr) {
      SENode* coefficient = child->AsSERelwrrentNode()->GetCoefficient();
      // If coefficient is zero then we can eliminate the relwrrent expression
      // entirely and just return the offset as the relwrrent expression is
      // representing the equation coefficient*iterations + offset.
      if (coefficient->GetType() == SENode::Constant &&
          coefficient->AsSEConstantNode()->FoldToSingleValue() == 0) {
        new_children.push_back(child->AsSERelwrrentNode()->GetOffset());
        has_change = true;
      } else {
        new_children.push_back(child);
      }
    } else {
      new_children.push_back(child);
    }
  }

  if (!has_change) return node;

  std::unique_ptr<SENode> new_add{new SEAddNode(node_->GetParentAnalysis())};

  for (SENode* child : new_children) {
    new_add->AddChild(child);
  }

  return analysis_.GetCachedOrAdd(std::move(new_add));
}

SENode* SENodeSimplifyImpl::SimplifyRelwrrentAddExpression(
    SERelwrrentNode* relwrrent_expr) {
  const std::vector<SENode*>& children = node_->GetChildren();

  std::unique_ptr<SERelwrrentNode> relwrrent_node{new SERelwrrentNode(
      relwrrent_expr->GetParentAnalysis(), relwrrent_expr->GetLoop())};

  // Create and simplify the new offset node.
  std::unique_ptr<SENode> new_offset{
      new SEAddNode(relwrrent_expr->GetParentAnalysis())};
  new_offset->AddChild(relwrrent_expr->GetOffset());

  for (SENode* child : children) {
    if (child->GetType() != SENode::RelwrrentAddExpr) {
      new_offset->AddChild(child);
    }
  }

  // Simplify the new offset.
  SENode* simplified_child = analysis_.SimplifyExpression(new_offset.get());

  // If the child can be simplified, add the simplified form otherwise, add it
  // via the usual caching mechanism.
  if (simplified_child->GetType() != SENode::CanNotCompute) {
    relwrrent_node->AddOffset(simplified_child);
  } else {
    relwrrent_expr->AddOffset(analysis_.GetCachedOrAdd(std::move(new_offset)));
  }

  relwrrent_node->AddCoefficient(relwrrent_expr->GetCoefficient());

  return analysis_.GetCachedOrAdd(std::move(relwrrent_node));
}

/*
 * Scalar Analysis simplification public methods.
 */

SENode* ScalarEvolutionAnalysis::SimplifyExpression(SENode* node) {
  SENodeSimplifyImpl impl{this, node};

  return impl.Simplify();
}

}  // namespace opt
}  // namespace spvtools
