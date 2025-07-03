<!-- XSLT script to colwert doxygen class xml to html -->
<xsl:stylesheet xmlns:xsl="http://www.w3.org/1999/XSL/Transform" version="1.0">
  <xsl:template match="/">
  <html>
  <body>
    <!-- Load all doxgen generated xml files -->
    <xsl:for-each select="//compounddef[@kind='class']">
      <!-- Generate class name with template parameters if any -->
      <xsl:variable name="class-name" disable-output-escaping="yes">
        <xsl:value-of select="compoundname"/>
        <xsl:if test="templateparamlist">
         <xsl:for-each select="templateparamlist/param">
            <xsl:choose>
             <xsl:when test="position() = 1">&lt;</xsl:when>
           </xsl:choose>
           <xsl:value-of select="type"/>
           <xsl:if test="declname">
             <xsl:text> </xsl:text>
             <xsl:value-of select="declname"/>
           </xsl:if>
           <xsl:choose>
             <xsl:when test="position() = last()">&gt;</xsl:when>
             <xsl:otherwise><xsl:text>, </xsl:text></xsl:otherwise>
           </xsl:choose>
         </xsl:for-each>
        </xsl:if>
      </xsl:variable>

      <h3 style="color:#76b900;">class <xsl:value-of select="$class-name"/></h3>

      <xsl:for-each select="briefdescription/para">
        <p style="margin-left:40px"><xsl:value-of select="."/></p>
      </xsl:for-each>

      <xsl:for-each select="detaileddescription/para">
        <xsl:if test="text()">
          <p style="margin-left:40px"><xsl:value-of select='./text()[normalize-space()][1]'/></p>
        </xsl:if>

        <xsl:if test="itemizedlist/listitem">
          <xsl:for-each select="itemizedlist/listitem/para">
            <ul style="margin-left:80px">
              <li>
                <xsl:value-of select="."/>
              </li>
            </ul>
          </xsl:for-each>
        </xsl:if>

        <xsl:if test="parameterlist[@kind='templateparam']">
          <br/><h4 style="margin-left:40px;color:#76b900;">Template Parameters</h4>
        </xsl:if>

        <xsl:for-each select="parameterlist[@kind='templateparam']/parameteritem">
          <ul style="margin-left:80px">
            <li>
              <b><xsl:value-of select="parameternamelist/parametername"/></b>
              <xsl:text>: </xsl:text>
              <xsl:choose>
                <xsl:when test="parameterdescription/para">
                  <xsl:value-of select="parameterdescription/para/node()[not(self::itemizedlist)]"/>
                </xsl:when>
                <xsl:otherwise>
                  <xsl:value-of select="parameterdescription/para/."/>
                </xsl:otherwise>
              </xsl:choose>
              <xsl:for-each select="parameterdescription/para/itemizedlist/listitem">
                <ul style="margin-left:40px">
                  <li>
                    <xsl:value-of select="."/>
                  </li>
                </ul>
              </xsl:for-each>
            </li>
          </ul>
        </xsl:for-each>
      </xsl:for-each>

      <xsl:if test="sectiondef[@kind='public-func']/memberdef[@kind='function' and @virt='pure-virtual']">
        <h4 style="margin-left:40px;color:#76b900;">Public Functions</h4>
      </xsl:if>
      <p style="margin-left:40px">
        <xsl:for-each select="sectiondef[@kind='public-func']/memberdef[@kind='function' and @virt='pure-virtual']">
          <p style="margin-left:40px">
              <!-- <xsl:variable name="def" select="definition"/>
              <xsl:if test="contains($def,'virtual')">
                virtual
              </xsl:if>
              <xsl:if test="contains($def,'static')">
                static
              </xsl:if> -->
              <xsl:value-of select="type"/>
              <xsl:text> </xsl:text>
              <b><xsl:value-of select="name"/></b>
              <!-- TODO: Format argstring below by replacing it <param siblings> -->
              <xsl:value-of select="argsstring"/>
          </p>
          <p style="margin-left:80px">
            <xsl:if test="briefdescription/para">
              <p style="margin-left:80px"><xsl:value-of select="briefdescription/para"/></p>
            </xsl:if>
            <xsl:for-each select="detaileddescription/para">
              <xsl:if test="not(parameterlist)">
                <p style="margin-left:80px"><xsl:value-of select="."/></p>
              </xsl:if>
              <xsl:if test="parameterlist[@kind='param']">
                <p style="margin-left:80px"><b>Parameters</b></p>
              </xsl:if>
              <xsl:for-each select="parameterlist[@kind='param']/parameteritem">
                <ul style="margin-left:120px">
                  <li>
                    [<xsl:value-of select="parameternamelist/parametername/@direction"/>]
                    <xsl:text> </xsl:text>
                    <xsl:value-of select="parameternamelist/parametername"/>
                    <xsl:text>: </xsl:text>
                    <xsl:value-of select="parameterdescription/para"/>
                  </li>
                </ul>
              </xsl:for-each>
              <xsl:if test="simplesect[@kind='return']">
                <p style="margin-left:80px">
                  <b>Returns</b><xsl:text> </xsl:text>
                  <br/>
                  <xsl:value-of select="simplesect[@kind='return']/para/node()[not(self::itemizedlist)]"/>
                  <br/>
                  <xsl:for-each select="simplesect/para/itemizedlist/listitem">
                    <ul style="margin-left:120px">
                      <li>
                        <xsl:choose>
                          <xsl:when test="para/itemizedlist">
                            <xsl:value-of select="para/node()[not(self::itemizedlist)]"/>
                          </xsl:when>
                          <xsl:otherwise>
                            <xsl:value-of select="para/."/>
                          </xsl:otherwise>
                        </xsl:choose>
                        <!-- <xsl:value-of select="para/."/> -->
                        <xsl:for-each select="para/itemizedlist/listitem">
                          <ul style="margin-left:140px">
                            <li>
                              <xsl:value-of select="."/>
                            </li>
                          </ul>
                        </xsl:for-each>
                      </li>
                    </ul>
                  </xsl:for-each>
                </p>
              </xsl:if>
            </xsl:for-each>
          </p>
        </xsl:for-each>
      </p>

      <xsl:if test="sectiondef[@kind='private-type']/memberdef[@kind='typedef']">
        <xsl:variable name="flag">
         <xsl:value-of select="'0'"/>
         <xsl:for-each select="sectiondef[@kind='private-type']/memberdef[@kind='typedef']">
          <xsl:value-of select="briefdescription/para"/>
          <xsl:value-of select="detaileddescription/para"/>
         </xsl:for-each>
        </xsl:variable>

        <xsl:if test="0 != $flag">
          <br/><h4 style="margin-left:40px;color:#76b900;">Private Types</h4>
        </xsl:if>
      </xsl:if>

      <xsl:for-each select="sectiondef[@kind='private-type']/memberdef[@kind='typedef']">
        <xsl:variable name="brief-desc" select="briefdescription/para"/>
        <xsl:variable name="detail-desc" select="detaileddescription/para"/>
        <xsl:if test="$brief-desc or $detail-desc">
          <p style="margin-left:40px">
              <b><xsl:value-of select="definition"/></b>
              <p style="margin-left:80px">
                <xsl:value-of select="briefdescription/para"/>
                <xsl:if test="not(briefdescription/para)">
                  <xsl:value-of select="detaileddescription/para"/>
                </xsl:if>
              </p>
          </p><br/>
        </xsl:if>
      </xsl:for-each>

      <xsl:if test="sectiondef[@kind='protected-type']/memberdef[@kind='typedef']">
        <br/><h4 style="margin-left:40px;color:#76b900;">Protected Types</h4>
      </xsl:if>
      <xsl:for-each select="sectiondef[@kind='protected-type']/memberdef[@kind='typedef']">
          <p style="margin-left:40px">
              <b><xsl:value-of select="definition"/></b>
              <p style="margin-left:80px">
                <xsl:value-of select="briefdescription/para"/>
                <xsl:if test="not(briefdescription/para)">
                  <xsl:value-of select="detaileddescription/para"/>
                </xsl:if>
              </p>
          </p><br/>
      </xsl:for-each>

      <xsl:if test="sectiondef[@kind='public-static-attrib']/memberdef[@kind='variable']">
        <br/><h4 style="margin-left:40px;color:#76b900;">Public Static Attributes</h4>
      </xsl:if>
      <xsl:for-each select="sectiondef[@kind='public-static-attrib']/memberdef[@kind='variable']">
        <p style="margin-left:40px">
            static
            <xsl:text> </xsl:text>
            <xsl:value-of select="type"/>
            <xsl:text> </xsl:text>
            <b><xsl:value-of select="name"/></b>
            <xsl:text> </xsl:text>
            <xsl:value-of select="initializer"/>
            <p style="margin-left:80px">
              <xsl:value-of select="briefdescription/para"/>
              <xsl:if test="not(briefdescription/para)">
                <xsl:value-of select="detaileddescription/para"/>
              </xsl:if>
            </p>
        </p><br/>
      </xsl:for-each>

      <xsl:if test="sectiondef[@kind='private-attrib']/memberdef[@kind='variable']">
        <xsl:variable name="flag">
         <xsl:value-of select="'0'"/>
         <xsl:for-each select="sectiondef[@kind='private-attrib']/memberdef[@kind='variable']">
          <xsl:value-of select="briefdescription/para"/>
          <xsl:value-of select="detaileddescription/para"/>
         </xsl:for-each>
        </xsl:variable>

        <xsl:if test="0 != $flag">
         <br/><h4 style="margin-left:40px;color:#76b900;">Private Attributes</h4>
        </xsl:if>
      </xsl:if>

      <xsl:for-each select="sectiondef[@kind='private-attrib']/memberdef[@kind='variable']">
        <xsl:variable name="brief-desc" select="briefdescription/para"/>
        <xsl:variable name="detail-desc" select="detaileddescription/para"/>
        <xsl:if test="$brief-desc or $detail-desc">
          <p style="margin-left:40px">
              <xsl:value-of select="type"/>
              <xsl:text> </xsl:text>
              <b><xsl:value-of select="name"/></b>
              <p style="margin-left:80px">
                <xsl:value-of select="briefdescription/para"/>
                <xsl:if test="not(briefdescription/para)">
                  <xsl:value-of select="detaileddescription/para"/>
                </xsl:if>
              </p>
          </p><br/>
        </xsl:if>
      </xsl:for-each>

      <xsl:if test="sectiondef[@kind='public-attrib']/memberdef[@kind='variable']">
        <br/><h4 style="margin-left:40px;color:#76b900;">Public Attributes</h4>
      </xsl:if>
      <xsl:for-each select="sectiondef[@kind='public-attrib']/memberdef[@kind='variable']">
         <p style="margin-left:40px">
            <xsl:value-of select="type"/>
            <xsl:text> </xsl:text>
            <b><xsl:value-of select="name"/></b>
            <p style="margin-left:80px">
              <xsl:value-of select="briefdescription/para"/>
              <xsl:if test="not(briefdescription/para)">
                <xsl:value-of select="detaileddescription/para"/>
              </xsl:if>
            </p>
         </p><br/>
      </xsl:for-each>

      <xsl:if test="sectiondef[@kind='private-static-attrib']/memberdef[@kind='variable']">
        <xsl:variable name="flag">
          <xsl:value-of select="'0'"/>
          <xsl:for-each select="sectiondef[@kind='private-static-attrib']/memberdef[@kind='variable']">
            <xsl:value-of select="briefdescription/para"/>
            <xsl:value-of select="detaileddescription/para"/>
          </xsl:for-each>
        </xsl:variable>

        <xsl:if test="0 != $flag">
          <br/><h4 style="margin-left:40px;color:#76b900;">Private Static Attributes</h4>
        </xsl:if>
      </xsl:if>

      <xsl:for-each select="sectiondef[@kind='private-static-attrib']/memberdef[@kind='variable']">
        <xsl:variable name="brief-desc" select="briefdescription/para"/>
        <xsl:variable name="detail-desc" select="detaileddescription/para"/>
        <xsl:if test="$brief-desc or $detail-desc">
          <p style="margin-left:40px">
              static
              <xsl:value-of select="type"/>
              <xsl:text> </xsl:text>
              <b><xsl:value-of select="name"/></b>
              <xsl:text> </xsl:text>
              <xsl:value-of select="initializer"/>
              <p style="margin-left:80px">
                <xsl:value-of select="briefdescription/para"/>
                <xsl:if test="not(briefdescription/para)">
                  <xsl:value-of select="detaileddescription/para"/>
                </xsl:if>
              </p>
          </p><br/>
        </xsl:if>
      </xsl:for-each>

      <xsl:if test="sectiondef[@kind='protected-attrib']/memberdef[@kind='variable']">
        <br/><h4 style="margin-left:40px;color:#76b900;">Protected Attributes</h4>
      </xsl:if>
      <xsl:for-each select="sectiondef[@kind='protected-attrib']/memberdef[@kind='variable']">
        <p style="margin-left:40px">
            <xsl:value-of select="type"/>
            <xsl:text> </xsl:text>
            <b><xsl:value-of select="name"/></b>
            <p style="margin-left:80px">
              <xsl:value-of select="briefdescription/para"/>
              <xsl:if test="not(briefdescription/para)">
                <xsl:value-of select="detaileddescription/para"/>
              </xsl:if>
            </p>
        </p><br/>
      </xsl:for-each>

      <xsl:if test="sectiondef[@kind='protected-static-attrib']/memberdef[@kind='variable']">
        <br/><h4 style="margin-left:40px;color:#76b900;">Protected Static Attributes</h4>
      </xsl:if>
      <xsl:for-each select="sectiondef[@kind='protected-static-attrib']/memberdef[@kind='variable']">
          <p style="margin-left:40px">
              static
              <xsl:value-of select="type"/>
              <xsl:text> </xsl:text>
              <b><xsl:value-of select="name"/></b>
              <xsl:text> </xsl:text>
              <xsl:value-of select="initializer"/>
              <p style="margin-left:80px">
                <xsl:value-of select="briefdescription/para"/>
                <xsl:if test="not(briefdescription/para)">
                  <xsl:value-of select="detaileddescription/para"/>
                </xsl:if>
              </p>
          </p><br/>
      </xsl:for-each>

      <h4 style="margin-left:40px;color:#76b900;">File Location</h4>
      <p style="margin-left:80px">
        <xsl:value-of select="location/@file"/>
      </p>
    </xsl:for-each>
  </body>
  </html>
  </xsl:template>
</xsl:stylesheet>
