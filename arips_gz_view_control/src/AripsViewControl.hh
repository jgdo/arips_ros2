/*
 * Copyright (C) 2021 Open Source Robotics Foundation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
*/

#ifndef ARIPS_GZ_VIEW_CONTROL_ARIPSVIEWCONTROL_HH_
#define ARIPS_GZ_VIEW_CONTROL_ARIPSVIEWCONTROL_HH_

#include <memory>

#include "gz/gui/Plugin.hh"

#include <gz/utils/ImplPtr.hh>

namespace arips_gz_view_control
{
  /// \brief Copy of gz-gui 10.1.0's InteractiveViewControl with the left and
  /// middle mouse buttons swapped. Controls a user camera with the mouse:
  ///
  /// * Drag left button to orbit (shift + left button to pan)
  /// * Drag middle button to pan
  /// * Drag right button or scroll wheel to zoom
  ///
  /// This plugin also supports changing between perspective and orthographic
  /// projections through the `/gui/camera/view_control` service. Perspective
  /// projection is used by default. For example:
  ///
  ///     gz service -s /gui/camera/view_control
  ///         --reqtype gz.msgs.StringMsg
  ///         --reptype gz.msgs.Boolean
  ///         --timeout 2000 --req 'data: "ortho"'
  ///
  /// Supported options are:
  ///
  /// * `orbit`: perspective projection
  /// * `ortho`: orthographic projection
  class AripsViewControl : public gz::gui::Plugin
  {
    Q_OBJECT

    /// \brief Constructor
    public: AripsViewControl();

    /// \brief Destructor
    public: virtual ~AripsViewControl();

    // Documentation inherited
    public: virtual void LoadConfig(const tinyxml2::XMLElement *_pluginElem)
        override;

    // Documentation inherited
    private: bool eventFilter(QObject *_obj, QEvent *_event) override;

    /// \internal
    /// \brief Pointer to private data.
    GZ_UTILS_UNIQUE_IMPL_PTR(dataPtr)
  };
}  // namespace arips_gz_view_control

#endif  // ARIPS_GZ_VIEW_CONTROL_ARIPSVIEWCONTROL_HH_
